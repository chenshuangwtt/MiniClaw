"""Tests for tool_registry.py."""

import pytest
from miniclaw.tool_registry import ToolRegistry


class TestToolRegistry:
    def test_register_and_list(self):
        reg = ToolRegistry()

        @reg.register(name="greet", description="Say hello.")
        def greet(name: str) -> str:
            return f"Hello, {name}!"

        assert "greet" in reg
        assert reg.list_tools() == ["greet"]
        assert len(reg) == 1

    def test_execute(self):
        reg = ToolRegistry()

        @reg.register(name="add", description="Add two numbers.")
        def add(a: int, b: int) -> int:
            return a + b

        result = reg.execute("add", {"a": 3, "b": 4})
        assert result == 7

    def test_execute_unknown_tool_raises(self):
        reg = ToolRegistry()
        with pytest.raises(KeyError, match="not registered"):
            reg.execute("nope", {})

    def test_get_returns_tool_def(self):
        reg = ToolRegistry()

        @reg.register(name="echo", description="Echo.")
        def echo(text: str) -> str:
            return text

        tool = reg.get("echo")
        assert tool is not None
        assert tool.name == "echo"
        assert tool.description == "Echo."

    def test_get_unknown_returns_none(self):
        reg = ToolRegistry()
        assert reg.get("missing") is None

    def test_to_openai_tools(self):
        reg = ToolRegistry()

        @reg.register(
            name="weather",
            description="Get weather.",
            parameters={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
        def weather(city: str) -> str:
            return f"Sunny in {city}"

        schema = reg.to_openai_tools()
        assert len(schema) == 1
        assert schema[0]["type"] == "function"
        assert schema[0]["function"]["name"] == "weather"

    def test_infer_parameters_from_annotations(self):
        reg = ToolRegistry()

        @reg.register(name="calc", description="Calculate.")
        def calc(x: int, y: float = 1.0) -> float:
            return x * y

        tool = reg.get("calc")
        params = tool.parameters
        assert params["properties"]["x"]["type"] == "integer"
        assert params["properties"]["y"]["type"] == "number"
        assert "x" in params["required"]
        assert "y" not in params["required"]

    def test_register_without_name_uses_function_name(self):
        reg = ToolRegistry()

        @reg.register(description="Does something.")
        def my_tool() -> str:
            return "done"

        assert "my_tool" in reg
