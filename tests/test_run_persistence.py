"""Tests for durable run state and checkpoint semantics."""

from __future__ import annotations

from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.fake import FakeLLM
from miniclaw.storage.sqlite_store import SQLiteStore
from miniclaw.tools.base import Tool, ToolCapability
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry


class WriteProbe(Tool):
    name = "write_probe"
    description = "Test-only write tool."
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self) -> str:
        return "written"


def test_completed_run_and_checkpoint_are_persisted() -> None:
    registry = ToolRegistry()
    with SQLiteStore(":memory:") as store:
        agent = AgentLoop(
            llm=FakeLLM(['{"type": "final_answer", "answer": "done"}']),
            registry=registry,
            store=store,
        )
        result = agent.run("finish")

        run = store.get_run(result.run_id)
        checkpoint = store.get_latest_checkpoint(result.run_id)

    assert result.success is True
    assert run is not None and run["status"] == "completed"
    assert checkpoint is not None
    assert checkpoint["state"]["task"] == "finish"


def test_approval_wait_is_durable_and_does_not_execute_tool() -> None:
    registry = ToolRegistry()
    tool = WriteProbe()
    registry.register(tool)
    tool.capability = ToolCapability.FILE_WRITE
    policy = PermissionPolicy(
        allow_file_write=True,
        approval_required_tools={"write_probe"},
    )
    with SQLiteStore(":memory:") as store:
        agent = AgentLoop(
            llm=FakeLLM(['{"type": "tool_call", "tool_name": "write_probe", "arguments": {}}']),
            registry=registry,
            permission_policy=policy,
            store=store,
        )
        result = agent.run("write only after approval")
        run = store.get_run(result.run_id)

    assert result.status.value == "waiting_approval"
    assert result.success is False
    assert run is not None and run["status"] == "waiting_approval"
