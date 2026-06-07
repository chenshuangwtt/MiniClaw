"""Tests for agent/recovery.py."""

import json

from miniclaw.agent.recovery import RecoveryManager
from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


# ============================================================
# Fixtures
# ============================================================


class EchoTool(Tool):
    name = "echo"
    description = "Echo text."
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string", "description": "Text to echo."}},
        "required": ["text"],
    }

    def run(self, text: str, **kwargs):
        return text


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    def run(self, a: float, b: float, **kwargs):
        return a + b


def _make_registry() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    reg.register(AddTool())
    return reg


# ============================================================
# handle_invalid_json
# ============================================================


class TestHandleInvalidJson:
    def setup_method(self):
        self.rm = RecoveryManager()

    def test_valid_json_passthrough(self):
        raw = '{"type": "final_answer", "answer": "hi"}'
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "repaired"
        assert result["output"] == raw

    def test_json_embedded_in_text(self):
        raw = 'Here is my answer: {"type": "final_answer", "answer": "42"} hope that helps!'
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "repaired"
        obj = json.loads(result["output"])
        assert obj["type"] == "final_answer"
        assert obj["answer"] == "42"

    def test_nested_json_extracted(self):
        raw = (
            'Sure! {"type": "tool_call", "tool_name": "echo", "arguments": {"text": "hello"}} done.'
        )
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "repaired"
        obj = json.loads(result["output"])
        assert obj["tool_name"] == "echo"

    def test_no_json_returns_failed(self):
        raw = "I don't know what to say, here's some text."
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "failed"
        assert "not valid JSON" in result["error"]
        assert "tool_call" in result["error"]
        assert "final_answer" in result["error"]

    def test_empty_string_returns_failed(self):
        result = self.rm.handle_invalid_json("")
        assert result["status"] == "failed"

    def test_broken_json_returns_failed(self):
        raw = '{"type": "tool_call", broken...'
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "failed"

    def test_json_array_not_extracted(self):
        """Only objects are extracted, not arrays."""
        raw = "[1, 2, 3]"
        result = self.rm.handle_invalid_json(raw)
        # The first '{' is not found, so it fails
        assert result["status"] == "failed"

    def test_repaired_is_valid_json(self):
        raw = 'Blah blah {"type": "final_answer", "answer": "ok"} blah'
        result = self.rm.handle_invalid_json(raw)
        assert result["status"] == "repaired"
        # Should be parseable
        obj = json.loads(result["output"])
        assert isinstance(obj, dict)


# ============================================================
# handle_unknown_tool
# ============================================================


class TestHandleUnknownTool:
    def setup_method(self):
        self.rm = RecoveryManager()
        self.reg = _make_registry()

    def test_lists_available_tools(self):
        result = self.rm.handle_unknown_tool("ghost", self.reg)
        assert result["status"] == "error"
        assert "ghost" in result["error"]
        assert "echo" in result["error"]
        assert "add" in result["error"]

    def test_suggests_final_answer(self):
        result = self.rm.handle_unknown_tool("nope", self.reg)
        assert "final_answer" in result["error"]

    def test_empty_registry(self):
        empty = ToolRegistry()
        result = self.rm.handle_unknown_tool("x", empty)
        assert "no tools registered" in result["error"]


# ============================================================
# handle_bad_arguments
# ============================================================


class TestHandleBadArguments:
    def setup_method(self):
        self.rm = RecoveryManager()
        self.reg = _make_registry()

    def test_includes_error_message(self):
        result = self.rm.handle_bad_arguments("add", "missing required 'a'", self.reg)
        assert result["status"] == "error"
        assert "missing required 'a'" in result["error"]

    def test_includes_schema(self):
        result = self.rm.handle_bad_arguments("add", "bad type", self.reg)
        assert '"a"' in result["error"]
        assert '"b"' in result["error"]
        assert "number" in result["error"]

    def test_unknown_tool_schema_unavailable(self):
        result = self.rm.handle_bad_arguments("ghost", "err", self.reg)
        assert "unavailable" in result["error"]

    def test_includes_tool_name(self):
        result = self.rm.handle_bad_arguments("echo", "wrong", self.reg)
        assert "echo" in result["error"]


# ============================================================
# handle_tool_error
# ============================================================


class TestHandleToolError:
    def setup_method(self):
        self.rm = RecoveryManager()

    def test_includes_tool_name_and_error(self):
        result = self.rm.handle_tool_error("fail", "RuntimeError: kaboom")
        assert result["status"] == "error"
        assert "fail" in result["error"]
        assert "kaboom" in result["error"]

    def test_suggests_alternatives(self):
        result = self.rm.handle_tool_error("x", "oops")
        assert "retry" in result["error"].lower() or "different" in result["error"].lower()

    def test_mentions_final_answer(self):
        result = self.rm.handle_tool_error("x", "err")
        assert "final_answer" in result["error"]


# ============================================================
# handle_consecutive_failures
# ============================================================


class TestHandleConsecutiveFailures:
    def test_below_threshold_returns_none(self):
        rm = RecoveryManager(max_errors=3)
        assert rm.handle_consecutive_failures(2) is None

    def test_at_threshold_returns_abort(self):
        rm = RecoveryManager(max_errors=3)
        result = rm.handle_consecutive_failures(3)
        assert result is not None
        assert result["status"] == "abort"
        assert result["type"] == "final_answer"
        assert "3 consecutive errors" in result["answer"]

    def test_above_threshold_returns_abort(self):
        rm = RecoveryManager(max_errors=2)
        result = rm.handle_consecutive_failures(5)
        assert result is not None
        assert "5 consecutive errors" in result["answer"]

    def test_abort_message_is_model_friendly(self):
        rm = RecoveryManager(max_errors=2)
        result = rm.handle_consecutive_failures(2)
        assert "rephrasing" in result["answer"].lower() or "smaller" in result["answer"].lower()

    def test_zero_errors_returns_none(self):
        rm = RecoveryManager(max_errors=3)
        assert rm.handle_consecutive_failures(0) is None


# ============================================================
# _extract_first_json (internal helper)
# ============================================================


class TestExtractFirstJson:
    def test_simple_json(self):
        result = RecoveryManager._extract_first_json('{"a": 1}')
        assert result == '{"a": 1}'

    def test_nested_json(self):
        result = RecoveryManager._extract_first_json('{"a": {"b": 2}}')
        obj = json.loads(result)
        assert obj["a"]["b"] == 2

    def test_json_with_surrounding_text(self):
        result = RecoveryManager._extract_first_json('blah {"x": 1} blah')
        assert result == '{"x": 1}'

    def test_no_json(self):
        assert RecoveryManager._extract_first_json("no json here") is None

    def test_unclosed_brace(self):
        assert RecoveryManager._extract_first_json('{"a": 1') is None

    def test_invalid_json_content(self):
        # Has matching braces but invalid JSON
        assert RecoveryManager._extract_first_json("{broken json}") is None

    def test_first_json_only(self):
        text = '{"a": 1} and {"b": 2}'
        result = RecoveryManager._extract_first_json(text)
        assert result == '{"a": 1}'

    def test_multiple_nested(self):
        text = '{"a": {"b": {"c": 3}}}'
        result = RecoveryManager._extract_first_json(text)
        obj = json.loads(result)
        assert obj["a"]["b"]["c"] == 3


# ============================================================
# RecoveryManager config
# ============================================================


class TestRecoveryManagerConfig:
    def test_default_max_errors(self):
        rm = RecoveryManager()
        assert rm.max_errors == 3

    def test_custom_max_errors(self):
        rm = RecoveryManager(max_errors=5)
        assert rm.max_errors == 5
