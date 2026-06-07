"""Tests for tools/base.py and tools/registry.py."""

import pytest
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


# ============================================================
# Concrete test tools
# ============================================================


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the input text."
    schema = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Text to echo."},
        },
        "required": ["text"],
    }

    def run(self, text: str, **kwargs: Any) -> str:
        return text


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    schema = {
        "type": "object",
        "properties": {
            "a": {"type": "number"},
            "b": {"type": "number"},
        },
        "required": ["a", "b"],
    }

    def run(self, a: float, b: float, **kwargs: Any) -> float:
        return a + b


class FailingTool(Tool):
    name = "fail"
    description = "Always raises an error."
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> None:
        raise RuntimeError("Tool intentionally failed.")


# ============================================================
# Tool base class tests
# ============================================================


class TestToolBase:
    def test_name_and_description(self):
        t = EchoTool()
        assert t.name == "echo"
        assert t.description == "Echo back the input text."

    def test_schema(self):
        t = EchoTool()
        assert t.schema["type"] == "object"
        assert "text" in t.schema["properties"]

    def test_run(self):
        t = EchoTool()
        assert t.run(text="hello") == "hello"

    def test_to_openai_schema(self):
        t = EchoTool()
        s = t.to_openai_schema()
        assert s["type"] == "function"
        assert s["function"]["name"] == "echo"
        assert s["function"]["description"] == "Echo back the input text."
        assert s["function"]["parameters"] == t.schema

    def test_repr(self):
        t = EchoTool()
        assert "echo" in repr(t)

    def test_abstract_cannot_instantiate(self):
        with pytest.raises(TypeError):
            Tool()  # type: ignore[abstract]


# ============================================================
# ToolRegistry tests
# ============================================================


class TestToolRegistry:
    def test_register_and_len(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert len(reg) == 1

    def test_register_multiple(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(AddTool())
        assert len(reg) == 2

    def test_register_duplicate_raises(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        with pytest.raises(ValueError, match="already registered"):
            reg.register(EchoTool())

    def test_register_non_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(TypeError, match="Expected a Tool instance"):
            reg.register("not a tool")  # type: ignore[arg-type]

    def test_get_existing(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        tool = reg.get("echo")
        assert tool is not None
        assert tool.name == "echo"

    def test_get_missing_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("nope") is None

    def test_list_sorted(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        reg.register(EchoTool())
        assert reg.list() == ["add", "echo"]

    def test_list_empty(self):
        reg = ToolRegistry()
        assert reg.list() == []

    def test_contains(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        assert "echo" in reg
        assert "nope" not in reg

    def test_get_schema(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        schema = reg.get_schema("echo")
        assert schema is not None
        assert schema["name"] == "echo"
        assert schema["description"] == "Echo back the input text."
        assert "text" in schema["parameters"]["properties"]

    def test_get_schema_missing(self):
        reg = ToolRegistry()
        assert reg.get_schema("nope") is None

    def test_execute(self):
        reg = ToolRegistry()
        reg.register(AddTool())
        result = reg.execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_execute_unknown_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.execute("nope", {})

    def test_execute_tool_error_propagates(self):
        reg = ToolRegistry()
        reg.register(FailingTool())
        with pytest.raises(RuntimeError, match="intentionally failed"):
            reg.execute("fail", {})

    def test_to_openai_tools(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        reg.register(AddTool())
        schemas = reg.to_openai_tools()
        assert len(schemas) == 2
        names = {s["function"]["name"] for s in schemas}
        assert names == {"echo", "add"}

    def test_repr(self):
        reg = ToolRegistry()
        reg.register(EchoTool())
        r = repr(reg)
        assert "echo" in r
        assert "ToolRegistry" in r
