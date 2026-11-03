"""Regression coverage for final runtime integration guarantees."""

from __future__ import annotations

from miniclaw.agent.config import load_config
from miniclaw.agent.context import ContextManager
from miniclaw.agent.loop import AgentLoop
from miniclaw.agent.trace import TraceLogger
from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall
from miniclaw.llm.fake import FakeLLM
from miniclaw.storage.sqlite_store import SQLiteStore
from miniclaw.tools.base import Tool
from miniclaw.tools.mcp_adapter import MCPClient
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
