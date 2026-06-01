"""Abstract base class for all LLM providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    """A single tool invocation requested by the LLM."""

    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class LLMResponse:
    """Unified response from any LLM provider."""

    content: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    raw: Any = None  # provider-specific raw response for debugging

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


class BaseLLM(ABC):
    """Interface that every LLM provider must implement."""

    @abstractmethod
    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat completion request.

        Args:
            messages: OpenAI-format message list.
            tools: Tool definitions in OpenAI function-calling schema, or None.
            temperature: Sampling temperature.

        Returns:
            LLMResponse with content and/or tool_calls.
        """
        ...

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Estimate token count for a message list.

        Default: rough char-based estimate. Providers may override with
        a real tokenizer (e.g. tiktoken).
        """
        total = 0
        for msg in messages:
            total += len(str(msg.get("content", ""))) // 4
            if "tool_calls" in msg:
                total += len(str(msg["tool_calls"])) // 4
        return total
