"""Tool registry — register, discover, and execute tools."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class ToolDef:
    """Definition of a single tool."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema for the parameters object
    fn: Callable[..., Any]

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


class ToolRegistry:
    """Central registry for all tools available to the agent."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolDef] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(
        self,
        name: str | None = None,
        description: str = "",
        parameters: dict[str, Any] | None = None,
    ) -> Callable:
        """Decorator to register a function as a tool.

        Usage::

            registry = ToolRegistry()

            @registry.register(
                name="get_weather",
                description="Get current weather for a city.",
                parameters={
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"}
                    },
                    "required": ["city"],
                },
            )
            def get_weather(city: str) -> str:
                return f"Sunny, 25°C in {city}"
        """

        def decorator(fn: Callable) -> Callable:
            tool_name = name or fn.__name__
            tool_def = ToolDef(
                name=tool_name,
                description=description or fn.__doc__ or "",
                parameters=parameters or _infer_parameters(fn),
                fn=fn,
            )
            self._tools[tool_name] = tool_def
            return fn

        return decorator

    # ------------------------------------------------------------------
    # Lookup & execution
    # ------------------------------------------------------------------

    def get(self, name: str) -> ToolDef | None:
        """Look up a tool by name, or return None."""
        return self._tools.get(name)

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by *name* with *arguments*.

        Raises:
            KeyError: if the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool.fn(**arguments)

    def list_tools(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Return all tool definitions in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools


# ------------------------------------------------------------------
# Helper: infer JSON Schema from function signature
# ------------------------------------------------------------------


def _infer_parameters(fn: Callable) -> dict[str, Any]:
    """Best-effort JSON Schema from a function's type annotations."""
    sig = inspect.signature(fn)
    hints = fn.__annotations__ if hasattr(fn, "__annotations__") else {}

    TYPE_MAP = {
        str: "string",
        int: "integer",
        float: "number",
        bool: "boolean",
        list: "array",
        dict: "object",
    }

    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name in ("self", "cls"):
            continue
        json_type = TYPE_MAP.get(hints.get(param_name, str), "string")
        prop: dict[str, Any] = {"type": json_type}
        if param.default is inspect.Parameter.empty:
            required.append(param_name)
        properties[param_name] = prop

    return {
        "type": "object",
        "properties": properties,
        "required": required,
    }
