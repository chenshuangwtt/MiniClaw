"""OpenAI-compatible LLM client.

Works with the official OpenAI API and any compatible endpoint
(Ollama, vLLM, LiteLLM, etc.) via *base_url* override.
"""

from __future__ import annotations

import json
from typing import Any

from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall


class OpenAILLM(BaseLLM):
    """Thin wrapper around the ``openai`` Python SDK."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for OpenAILLM. "
                "Install it with: pip install openai"
            ) from exc

        self._client = OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        completion = self._client.chat.completions.create(**kwargs)
        choice = completion.choices[0]
        message = choice.message

        tool_calls: list[ToolCall] = []
        if message.tool_calls:
            for tc in message.tool_calls:
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(
                    ToolCall(id=tc.id, name=tc.function.name, arguments=args)
                )

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            raw=completion,
        )

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Use tiktoken for accurate token counting when available."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            return super().count_tokens(messages)

        total = 0
        for msg in messages:
            # rough: every message has ~4 tokens overhead
            total += 4
            content = msg.get("content") or ""
            total += len(enc.encode(content))
        return total
