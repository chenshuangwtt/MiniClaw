"""FakeLLM — a programmable mock for offline development and testing."""

from __future__ import annotations

import uuid
from typing import Any

from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall


class FakeLLM(BaseLLM):
    """Returns pre-scripted responses in order.

    Usage::

        llm = FakeLLM([
            "I'll check the weather for you.",
            '{"tool_call": {"name": "get_weather", "arguments": {"city": "Beijing"}}}',
            "The weather in Beijing is sunny, 25°C.",
        ])
        resp1 = llm.chat(messages)  # → first scripted response
        resp2 = llm.chat(messages)  # → second
        resp3 = llm.chat(messages)  # → third

    If a scripted string is valid JSON with a ``tool_call`` key, it is parsed
    into a ``ToolCall`` automatically.  Otherwise the string becomes
    ``LLMResponse.content``.
    """

    def __init__(self, scripted_responses: list[str | LLMResponse]) -> None:
        self._responses = list(scripted_responses)
        self._index = 0
        self.call_log: list[dict[str, Any]] = []

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        self.call_log.append({
            "messages": messages,
            "tools": tools,
            "temperature": temperature,
        })

        if self._index >= len(self._responses):
            return LLMResponse(content="[FakeLLM] No more scripted responses.")

        raw = self._responses[self._index]
        self._index += 1

        # If already an LLMResponse, return as-is
        if isinstance(raw, LLMResponse):
            return raw

        return self._parse_scripted(raw)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_scripted(text: str) -> LLMResponse:
        """Try to interpret *text* as a tool-call JSON or plain content."""
        import json

        text = text.strip()
        if not text:
            return LLMResponse(content="")

        try:
            obj = json.loads(text)
        except json.JSONDecodeError:
            return LLMResponse(content=text)

        if isinstance(obj, dict) and "tool_call" in obj:
            tc = obj["tool_call"]
            return LLMResponse(
                content=obj.get("content", ""),
                tool_calls=[
                    ToolCall(
                        id=f"call_{uuid.uuid4().hex[:8]}",
                        name=tc["name"],
                        arguments=tc.get("arguments", {}),
                    )
                ],
            )

        return LLMResponse(content=text)

    # ------------------------------------------------------------------
    # Convenience for tests
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the response index to the beginning."""
        self._index = 0

    @property
    def exhausted(self) -> bool:
        """True when all scripted responses have been consumed."""
        return self._index >= len(self._responses)
