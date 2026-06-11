"""Abstract base class for tools."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Tool(ABC):
    """Base class that every tool must subclass.

    A tool has four properties:
        - ``name``: unique identifier used by the LLM to reference this tool.
        - ``description``: human-readable text explaining what the tool does.
        - ``schema``: JSON Schema describing the tool's input parameters.
        - ``run()``: executes the tool with given arguments and returns a result.

    Optional attributes:
        - ``timeout``: default timeout in seconds.  If ``None``, the tool
          runs without a time limit (unless the caller imposes one).

    Example::

        class GetWeather(Tool):
            name = "get_weather"
            description = "Get the current weather for a city."
            schema = {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "City name"}
                },
                "required": ["city"],
            }

            def run(self, city: str) -> str:
                return f"Sunny, 25°C in {city}"
    """

    name: str
    description: str
    schema: dict[str, Any]
    timeout: float | None = None

    @abstractmethod
    def run(self, **kwargs: Any) -> Any:
        """Execute the tool with the given arguments.

        Args:
            **kwargs: Arguments validated against ``self.schema``.

        Returns:
            The tool's output (will be converted to string for the LLM).
        """
        ...

    def to_openai_schema(self) -> dict[str, Any]:
        """Return the tool definition in OpenAI function-calling format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.schema,
            },
        }

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"
