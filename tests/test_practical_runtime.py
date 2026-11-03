"""Offline end-to-end coverage for the practical runtime guarantees."""

from __future__ import annotations

import asyncio

from miniclaw.agent.context import ContextManager
from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.fake import FakeLLM
from miniclaw.memory.sqlite_backend import SQLiteMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor
from miniclaw.storage.sqlite_store import SQLiteStore
from miniclaw.tools.base import Tool, ToolCapability
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry


class EchoTool(Tool):
    name = "echo"
    description = "Return a repeatable result."
    schema = {
        "type": "object",
        "properties": {"value": {"type": "integer"}},
        "required": ["value"],
        "additionalProperties": False,
    }

    def run(self, value: int) -> dict[str, str]:
        return {"result": f"value={value}; " + "x" * 600}


class WriteOnceTool(Tool):
    name = "write_once"
    description = "Test-only non-idempotent write."
    capability = ToolCapability.FILE_WRITE
    schema = {"type": "object", "properties": {}, "required": []}

    def __init__(self) -> None:
        self.calls = 0

    def run(self) -> str:
        self.calls += 1
        return "written"


def test_cross_session_memory_is_selective_scoped_and_connected_to_agent() -> None:
    with SQLiteStore(":memory:") as store:
        backend = SQLiteMemoryBackend(store, project_id="project-a")
        first = AgentLoop(
            llm=FakeLLM(['{"type": "final_answer", "answer": "saved"}']),
            registry=ToolRegistry(),
            memory_backend=backend,
        )
        assert first.run("请记住，我偏好简洁的中文回答。", user_id="alice").success
        assert backend.search("偏好", "alice") == ["请记住，我偏好简洁的中文回答。"]
        stored = store.search_agent_memories("偏好", "alice", "project-a")
        assert stored[0]["source"] == "agent"
        assert backend.search("偏好", "bob") == []
        assert SQLiteMemoryBackend(store, project_id="project-b").search("偏好", "alice") == []

        second_llm = FakeLLM(['{"type": "final_answer", "answer": "used memory"}'])
        second = AgentLoop(
            llm=second_llm,
            registry=ToolRegistry(),
            memory_backend=backend,
            memory_extractor=MemoryExtractor(keywords=["never-store"]),
        )
        assert second.run("请根据偏好回答", user_id="alice").success
        assert "简洁的中文回答" in second_llm.call_log[0]["prompt"]

        assert backend.remove("请记住，我偏好简洁的中文回答。", "alice") is True
        assert backend.search("偏好", "alice") == []


def test_sensitive_memory_candidate_is_not_persisted() -> None:
    with SQLiteStore(":memory:") as store:
        backend = SQLiteMemoryBackend(store, project_id="project-a")
        result = AgentLoop(
            llm=FakeLLM(['{"type":"final_answer","answer":"done"}']),
            registry=ToolRegistry(),
            memory_backend=backend,
        ).run("请记住 token=super-secret-value", user_id="alice")

        assert result.success is True
        assert store.search_agent_memories("secret", "alice", "project-a") == []


def test_long_task_compresses_context_keeps_summary_and_completes() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    responses = [
        f'{{"type":"tool_call","tool_name":"echo","arguments":{{"value":{index}}}}}'
        for index in range(21)
    ]
    responses.append('{"type":"final_answer","answer":"all 21 tool results processed"}')
    llm = FakeLLM(responses)
    context = ContextManager(max_context_tokens=100, recent_keep=4, max_tool_result_chars=120)
    result = AgentLoop(
        llm=llm,
        registry=registry,
        context=context,
        max_steps=25,
        max_tool_calls=25,
    ).run("Complete the original 21-step audit task")

    assert result.success is True
    assert result.steps_taken == 22
    assert any(message["role"] == "summary" for message in context.get_messages())
    assert any("[Compressed" in call["prompt"] for call in llm.call_log[5:])
    tool_contents = [m["content"] for m in context.get_messages() if m["role"] == "tool"]
    assert all(len(content) <= 180 for content in tool_contents)


def test_cancelled_run_resumes_from_checkpoint_in_a_new_agent_instance() -> None:
    with SQLiteStore(":memory:") as store:
        original = AgentLoop(
            llm=FakeLLM(['{"type": "final_answer", "answer": "not consumed"}']),
            registry=ToolRegistry(),
            store=store,
        )
        original.cancel()
        cancelled = original.run("continue after interruption")
        assert cancelled.status.value == "cancelled"

        resumed = AgentLoop(
            llm=FakeLLM(['{"type": "final_answer", "answer": "resumed safely"}']),
            registry=ToolRegistry(),
            store=store,
        )
        result = asyncio.run(resumed.resume_async(cancelled.run_id))

    assert result.success is True
    assert result.answer == "resumed safely"


def test_approval_resume_runs_non_idempotent_tool_exactly_once() -> None:
    registry = ToolRegistry()
    writer = WriteOnceTool()
    registry.register(writer)
    policy = PermissionPolicy(allow_file_write=True, approval_required_tools={"write_once"})
    with SQLiteStore(":memory:") as store:
        paused = AgentLoop(
            llm=FakeLLM(['{"type":"tool_call","tool_name":"write_once","arguments":{}}']),
            registry=registry,
            permission_policy=policy,
            store=store,
        ).run("write only after approval")
        assert paused.status.value == "waiting_approval"
        assert writer.calls == 0

        resumed = AgentLoop(
            llm=FakeLLM(['{"type":"final_answer","answer":"write confirmed"}']),
            registry=registry,
            permission_policy=policy,
            store=store,
        )
        result = asyncio.run(resumed.resume_async(paused.run_id, approve_pending=True))

    assert result.success is True
    assert writer.calls == 1


def test_inflight_side_effect_checkpoint_is_not_replayed_automatically() -> None:
    registry = ToolRegistry()
    writer = WriteOnceTool()
    registry.register(writer)
    with SQLiteStore(":memory:") as store:
        store.create_run("inflight", "reconcile external write")
        store.update_run("inflight", "cancelled")
        store.save_checkpoint(
            "inflight",
            1,
            {
                "task": "reconcile external write",
                "mode": "portable",
                "context": [],
                "trace": [],
                "pending_action": {
                    "state": "in_flight",
                    "tool_name": "write_once",
                    "arguments": {},
                },
            },
        )
        result = asyncio.run(
            AgentLoop(
                llm=FakeLLM([]),
                registry=registry,
                permission_policy=PermissionPolicy(allow_file_write=True),
                store=store,
            ).resume_async("inflight", approve_pending=True)
        )

    assert result.success is False
    assert "manual reconciliation" in (result.error or "")
    assert writer.calls == 0


def test_repeat_budget_blocks_identical_tool_loop() -> None:
    registry = ToolRegistry()
    registry.register(EchoTool())
    llm = FakeLLM(
        [
            '{"type":"tool_call","tool_name":"echo","arguments":{"value":1}}',
            '{"type":"tool_call","tool_name":"echo","arguments":{"value":1}}',
            '{"type":"tool_call","tool_name":"echo","arguments":{"value":1}}',
            '{"type":"final_answer","answer":"should not hide the failure"}',
        ]
    )
    result = AgentLoop(
        llm=llm,
        registry=registry,
        max_repeated_tool_calls=2,
        max_errors=1,
    ).run("do not loop")

    assert result.success is False
    assert "Repeated identical" in (result.error or "")
