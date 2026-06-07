"""Tool registry — register, discover, and execute tools."""

from __future__ import annotations

from typing import Any

from miniclaw.tools.base import Tool


class ToolRegistry:
    """Central registry for all tools available to the agent.

    Usage::

        registry = ToolRegistry()
        registry.register(GetWeather())
        registry.register(Calculator())

        tool = registry.get("get_weather")
        result = tool.run(city="Beijing")
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: Tool) -> None:
        """Register a tool instance.

        Args:
            tool: A ``Tool`` subclass instance.

        Raises:
            TypeError: If *tool* is not a ``Tool`` instance.
            ValueError: If a tool with the same name is already registered.
        """
        if not isinstance(tool, Tool):
            raise TypeError(f"Expected a Tool instance, got {type(tool).__name__}.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get(self, name: str) -> Tool | None:
        """Look up a tool by name, or return ``None``."""
        return self._tools.get(name)

    def list(self) -> list[str]:
        """Return sorted list of registered tool names."""
        return sorted(self._tools.keys())

    def get_schema(self, name: str) -> dict[str, Any] | None:
        """Return the JSON Schema for a tool, or ``None`` if not found.

        The returned dict has the shape::

            {
                "name": "...",
                "description": "...",
                "parameters": { ... }   # JSON Schema
            }
        """
        tool = self._tools.get(name)
        if tool is None:
            return None
        return {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.schema,
        }

    def to_openai_tools(self) -> list[dict[str, Any]]:
        """Return all tool definitions in OpenAI function-calling format."""
        return [t.to_openai_schema() for t in self._tools.values()]

    # ------------------------------------------------------------------
    # Execution
    # ------------------------------------------------------------------

    def execute(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a registered tool by *name* with *arguments*.

        Args:
            name: Tool name.
            arguments: Keyword arguments to pass to ``tool.run()``.

        Returns:
            The tool's output.

        Raises:
            KeyError: If the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"Tool '{name}' is not registered.")
        return tool.run(**arguments)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._tools)

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __repr__(self) -> str:
        names = ", ".join(self.list())
        return f"<ToolRegistry tools=[{names}]>"
