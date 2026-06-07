"""Tests for agent/executor.py."""

from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry
from miniclaw.agent.executor import Observation, ToolExecutor
from miniclaw.agent.state import ToolCall


# ============================================================
# Test fixtures
# ============================================================


class EchoTool(Tool):
    name = "echo"
    description = "Echo."
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def run(self, text: str, **kwargs: Any) -> str:
        return text


class ExplodingTool(Tool):
    name = "explode"
    description = "Always fails."
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> None:
        raise ValueError("kaboom")


class AddTool(Tool):
    name = "add"
    description = "Add numbers."
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    def run(self, a: float, b: float, **kwargs: Any) -> float:
        return a + b


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(ExplodingTool())
    reg.register(AddTool())
    return reg


# ============================================================
# Observation model
# ============================================================


class TestObservation:
    def test_success_observation(self):
        obs = Observation(tool_name="echo", success=True, output="hi")
        assert obs.tool_name == "echo"
        assert obs.success is True
        assert obs.output == "hi"
        assert obs.error is None

    def test_failure_observation(self):
        obs = Observation(tool_name="explode", success=False, error="kaboom")
        assert obs.success is False
        assert obs.error == "kaboom"
        assert obs.output is None

    def test_default_values(self):
        obs = Observation(tool_name="x", success=True)
        assert obs.output is None
        assert obs.error is None


# ============================================================
# ToolExecutor
# ============================================================


class TestToolExecutor:
    def test_successful_execution(self):
        executor = ToolExecutor(_make_registry())
        obs = executor.execute("echo", {"text": "hello"})
        assert obs.success is True
        assert obs.output == "hello"
        assert obs.error is None

    def test_tool_error_captured(self):
        executor = ToolExecutor(_make_registry())
        obs = executor.execute("explode", {})
        assert obs.success is False
        assert "kaboom" in obs.error
        assert "ValueError" in obs.error

    def test_unknown_tool_never_raises(self):
        executor = ToolExecutor(_make_registry())
        obs = executor.execute("nonexistent", {"x": 1})
        assert obs.success is False
        assert "not registered" in obs.error.lower()

    def test_execute_never_raises(self):
        """The executor must never propagate exceptions."""
        executor = ToolExecutor(_make_registry())
        # All of these should return Observation, not raise
        obs1 = executor.execute("echo", {"text": "ok"})
        obs2 = executor.execute("explode", {})
        obs3 = executor.execute("missing", {})
        assert all(isinstance(o, Observation) for o in [obs1, obs2, obs3])

    def test_execute_with_arguments(self):
        executor = ToolExecutor(_make_registry())
        obs = executor.execute("add", {"a": 3, "b": 4})
        assert obs.success is True
        assert obs.output == 7

    def test_execute_tool_call_model(self):
        """Accept a ToolCall pydantic model directly."""
        executor = ToolExecutor(_make_registry())
        tc = ToolCall(tool_name="echo", arguments={"text": "from model"})
        obs = executor.execute_tool_call(tc)
        assert obs.success is True
        assert obs.output == "from model"

    def test_execute_tool_call_model_unknown(self):
        executor = ToolExecutor(_make_registry())
        tc = ToolCall(tool_name="ghost", arguments={})
        obs = executor.execute_tool_call(tc)
        assert obs.success is False
        assert "not registered" in obs.error.lower()

    def test_multiple_executions_independent(self):
        executor = ToolExecutor(_make_registry())
        obs1 = executor.execute("echo", {"text": "a"})
        obs2 = executor.execute("echo", {"text": "b"})
        assert obs1.output == "a"
        assert obs2.output == "b"
        assert obs1 is not obs2

    def test_permission_policy_blocks_tool(self):
        executor = ToolExecutor(
            _make_registry(),
            permission_policy=PermissionPolicy(approval_required_tools={"echo"}),
        )
        obs = executor.execute("echo", {"text": "blocked"})
        assert obs.success is False
        assert "requires approval" in obs.error

    def test_audit_logger_records_success(self):
        audit = AuditLogger()
        executor = ToolExecutor(_make_registry(), audit_logger=audit)
        obs = executor.execute("echo", {"text": "hello"})
        assert obs.success is True
        events = audit.events()
        assert len(events) == 1
        assert events[0].tool_name == "echo"
        assert events[0].allowed is True
        assert events[0].success is True

    def test_audit_logger_records_blocked_call(self):
        audit = AuditLogger()
        executor = ToolExecutor(
            _make_registry(),
            permission_policy=PermissionPolicy(approval_required_tools={"echo"}),
            audit_logger=audit,
        )
        obs = executor.execute("echo", {"text": "blocked"})
        assert obs.success is False
        events = audit.events()
        assert len(events) == 1
        assert events[0].allowed is False
        assert events[0].success is False
