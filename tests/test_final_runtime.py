"""Regression coverage for final runtime integration guarantees."""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

from miniclaw.agent.engine import AgentResult
from miniclaw.agent.config import Config, load_config
from miniclaw.agent.context import ContextManager
from miniclaw.agent.loop import AgentLoop
from miniclaw.agent.trace import TraceLogger
from miniclaw.agent.state import RunStatus
from miniclaw.cli import (
    EXIT_AGENT_FAILED,
    EXIT_SUCCESS,
    EXIT_WAITING_APPROVAL,
    _create_llm_from_config,
    _result_exit_code,
)
from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall
from miniclaw.llm.fake import FakeLLM
from miniclaw.redaction import redact_secrets
from miniclaw.storage.sqlite_store import SQLiteStore
from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.base import Tool, ToolCapability
from miniclaw.tools.mcp_adapter import MCPClient
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry


class NativeScript(BaseLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.requests: list[list[dict[str, object]]] = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.requests.append(messages)
        return self.responses.pop(0)


class Echo(Tool):
    name = "echo"
    description = "Echo text."
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, text: str) -> str:
        return text * 80


class SensitiveWrite(Tool):
    name = "sensitive_write"
    description = "Test-only sensitive write."
    capability = ToolCapability.FILE_WRITE
    schema = {
        "type": "object",
        "properties": {"api_key": {"type": "string"}},
        "required": ["api_key"],
        "additionalProperties": False,
    }

    def __init__(self) -> None:
        self.calls = 0

    def run(self, api_key: str) -> str:
        self.calls += 1
        return "written"


def test_dotenv_loads_compatible_provider_values_without_overriding_process_env(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.chdir(tmp_path)
    for name in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_MODEL"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text(
        "OPENAI_API_KEY=dotenv-test-key\nOPENAI_BASE_URL=https://example.invalid/v1\n"
        "DEEPSEEK_MODEL=deepseek-chat\n",
        encoding="utf-8",
    )
    config = load_config()
    assert config.llm.api_key == "dotenv-test-key"
    assert config.llm.base_url == "https://example.invalid/v1"
    assert config.llm.model == "deepseek-chat"

    monkeypatch.setenv("OPENAI_API_KEY", "process-wins")
    config = load_config()
    assert config.llm.api_key == "process-wins"


def test_portable_multiturn_context_keeps_prior_user_constraints_once() -> None:
    llm = FakeLLM(
        [
            '{"type":"final_answer","answer":"constraint noted"}',
            '{"type":"final_answer","answer":"task noted"}',
            '{"type":"final_answer","answer":"constraint applied"}',
        ]
    )
    agent = AgentLoop(llm=llm, registry=ToolRegistry())
    agent.run("Constraint: reply in Chinese and stay concise.")
    agent.run("Now inspect the plan.")
    result = agent.run("Apply the first constraint to the summary.")

    assert result.success
    prompt = llm.call_log[-1]["prompt"]
    assert "reply in Chinese and stay concise" in prompt
    assert prompt.count("Apply the first constraint to the summary.") == 1


def test_native_context_compression_preserves_tool_call_pair_and_bounds_result() -> None:
    registry = ToolRegistry()
    registry.register(Echo())
    llm = NativeScript(
        [
            LLMResponse(tool_calls=[ToolCall(id="call_1", name="echo", arguments={"text": "x"})]),
            LLMResponse(content="done"),
        ]
    )
    context = ContextManager(max_context_tokens=500, recent_keep=2, max_tool_result_chars=60)
    for index in range(20):
        context.add_message("user", f"Earlier discussion {index}: " + "detail " * 20)
    agent = AgentLoop(llm=llm, registry=registry, context=context)
    agent.parse_mode = "native"

    result = agent.run("Use echo, then answer.")

    assert result.success
    follow_up = llm.requests[1]
    assistant_index = next(i for i, message in enumerate(follow_up) if message.get("tool_calls"))
    tool_message = follow_up[assistant_index + 1]
    assert tool_message["role"] == "tool"
    assert tool_message["tool_call_id"] == "call_1"
    assert len(str(tool_message["content"])) <= 100


def test_trace_redacts_credentials_and_checkpoint_does_not_store_them() -> None:
    trace = TraceLogger()
    trace.log_step(
        1,
        model_output="token=very-secret-token-value",
        arguments={"Authorization": "Bearer very-secret-token-value"},
    )
    serialized = trace.steps[0].model_dump_json()
    assert "very-secret-token-value" not in serialized

    with SQLiteStore(":memory:") as store:
        agent = AgentLoop(
            llm=FakeLLM(['{"type":"final_answer","answer":"done"}']),
            registry=ToolRegistry(),
            store=store,
        )
        result = agent.run("token=very-secret-token-value")
        checkpoint = store.get_latest_checkpoint(result.run_id)
    assert checkpoint is not None
    assert "very-secret-token-value" not in str(checkpoint["state"])


def test_mcp_is_error_becomes_executor_failure_payload() -> None:
    client = MCPClient("unused")
    client._send_request = lambda method, params: {  # type: ignore[method-assign]
        "isError": True,
        "content": [{"type": "text", "text": "remote failure"}],
    }
    result = client.call_tool("broken", {})
    assert result["error"] == "MCP tool reported an error: remote failure"


def test_redaction_handles_nested_json_tool_arguments_without_mutating_input() -> None:
    native_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "example_tool",
                    "arguments": '{"api_key":"example-secret-value","path":"README.md"}',
                },
            }
        ],
        "nested": [{"Authorization": "Bearer example-secret-value-123456"}],
    }

    redacted = redact_secrets(native_message)
    arguments = json.loads(redacted["tool_calls"][0]["function"]["arguments"])
    assert arguments == {"api_key": "[REDACTED]", "path": "README.md"}
    assert redacted["nested"][0]["Authorization"] == "[REDACTED]"
    assert (
        native_message["tool_calls"][0]["function"]["arguments"]
        != redacted["tool_calls"][0]["function"]["arguments"]
    )
    assert redact_secrets(redacted) == redacted


def test_trace_audit_and_checkpoint_redact_native_arguments() -> None:
    native_message = {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "call_test",
                "type": "function",
                "function": {
                    "name": "example_tool",
                    "arguments": '{"token":"example-secret-value","path":"README.md"}',
                },
            }
        ],
    }
    trace = TraceLogger()
    trace.log_step(1, arguments=native_message)
    audit = AuditLogger()
    audit.log("example_tool", native_message, allowed=True, success=True)

    with SQLiteStore(":memory:") as store:
        store.create_run("run-1", "task")
        store.save_checkpoint("run-1", 1, redact_secrets({"context": [native_message]}))
        checkpoint = store.get_latest_checkpoint("run-1")

    for recorded in (trace.steps[0].arguments, audit.events()[0].arguments, checkpoint["state"]):
        assert "example-secret-value" not in str(recorded)
        assert (
            json.loads(recorded["tool_calls"][0]["function"]["arguments"])["path"] == "README.md"
            if "tool_calls" in recorded
            else True
        )


def test_cli_exit_code_mapping_and_real_process_exit_codes(tmp_path) -> None:
    trace = TraceLogger()
    assert (
        _result_exit_code(AgentResult(True, "ok", None, trace, 1, RunStatus.COMPLETED))
        == EXIT_SUCCESS
    )
    assert (
        _result_exit_code(AgentResult(False, None, "bad", trace, 1, RunStatus.FAILED))
        == EXIT_AGENT_FAILED
    )
    assert (
        _result_exit_code(AgentResult(False, None, "wait", trace, 1, RunStatus.WAITING_APPROVAL))
        == EXIT_WAITING_APPROVAL
    )

    repo_root = Path(__file__).resolve().parents[1]
    isolated_config = tmp_path / "missing-miniclaw.toml"
    (tmp_path / "README.md").write_text("# isolated CLI fixture\n", encoding="utf-8")
    (tmp_path / "src" / "miniclaw" / "agent").mkdir(parents=True)
    env = {key: value for key, value in os.environ.items() if not key.startswith("MINICLAW_")}
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "DEEPSEEK_API_KEY",
        "DEEPSEEK_BASE_URL",
        "DEEPSEEK_MODEL",
    ):
        env.pop(name, None)
    env.update(
        {
            "MINICLAW_LLM_PROVIDER": "fake",
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
        }
    )

    def run_cli(command: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=tmp_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="strict",
            env=env,
            check=False,
        )

    main_script = str(repo_root / "main.py")
    common_args = ["--config", str(isolated_config)]
    success = run_cli(
        [
            sys.executable,
            main_script,
            *common_args,
            "--db",
            str(tmp_path / "ok.db"),
            "run",
            "test task",
        ]
    )
    exhausted = run_cli(
        [
            sys.executable,
            main_script,
            *common_args,
            "--db",
            str(tmp_path / "failed.db"),
            "run",
            "--max-steps",
            "1",
            "test task",
        ]
    )
    invalid = run_cli([sys.executable, main_script, "--not-a-real-option"])

    console_name = "miniclaw.exe" if os.name == "nt" else "miniclaw"
    installed_cli = Path(sys.executable).with_name(console_name)
    assert installed_cli.exists()
    installed_success = run_cli(
        [str(installed_cli), *common_args, "--db", str(tmp_path / "console.db"), "run", "test task"]
    )
    installed_exhausted = run_cli(
        [
            str(installed_cli),
            *common_args,
            "--db",
            str(tmp_path / "console-failed.db"),
            "run",
            "--max-steps",
            "1",
            "test task",
        ]
    )
    assert success.returncode == EXIT_SUCCESS
    assert "任务：" in success.stdout
    assert exhausted.returncode == EXIT_AGENT_FAILED
    assert invalid.returncode == 2
    assert installed_success.returncode == EXIT_SUCCESS
    assert installed_exhausted.returncode == EXIT_AGENT_FAILED


def test_fake_llm_demo_describes_the_v1_runtime() -> None:
    llm = _create_llm_from_config(Config())
    for _ in range(4):
        llm.generate("demo")

    answer = json.loads(llm.generate("demo"))["answer"]

    for component in (
        "AgentEngine",
        "AgentLoop",
        "AsyncAgentLoop",
        "ToolRegistry / ToolExecutor",
        "ContextManager",
        "MemoryManager",
        "SQLiteStore",
        "Checkpoint",
        "TraceLogger",
    ):
        assert component in answer
    assert "确定性的 FakeLLM 演示脚本" in answer
    assert "四张表" not in answer


def test_sensitive_pending_checkpoint_is_redacted_and_never_replayed() -> None:
    registry = ToolRegistry()
    tool = SensitiveWrite()
    registry.register(tool)
    policy = PermissionPolicy(allow_file_write=True, approval_required_tools={tool.name})
    with SQLiteStore(":memory:") as store:
        paused = AgentLoop(
            llm=FakeLLM(
                [
                    '{"type":"tool_call","tool_name":"sensitive_write",'
                    '"arguments":{"api_key":"example-secret-value"}}'
                ]
            ),
            registry=registry,
            permission_policy=policy,
            store=store,
        ).run("wait for approval")
        checkpoint = store.get_latest_checkpoint(paused.run_id)
        assert checkpoint is not None
        assert "example-secret-value" not in str(checkpoint["state"])
        assert checkpoint["state"]["pending_action_reauth_required"] is True

        resumed = asyncio.run(
            AgentLoop(
                llm=FakeLLM(['{"type":"final_answer","answer":"should not run"}']),
                registry=registry,
                permission_policy=policy,
                store=store,
            ).resume_async(paused.run_id, approve_pending=True)
        )

    assert resumed.status is RunStatus.WAITING_APPROVAL
    assert "重新授权" in (resumed.error or "")
    assert tool.calls == 0
