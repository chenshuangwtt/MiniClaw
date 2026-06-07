"""OpenAI LLM client — reads API key from environment.

Provides a clean interface compatible with both ``BaseLLM`` (``chat``,
``generate``), a simplified ``call`` method, and **streaming** support.

Environment variables:
    ``OPENAI_API_KEY`` — required.  The OpenAI API key.
    ``OPENAI_BASE_URL`` — optional.  Override for compatible endpoints
        (Ollama, vLLM, LiteLLM, etc.).

Usage::

    from miniclaw.llm.openai_client import OpenAIClient

    # Uses OPENAI_API_KEY from environment
    llm = OpenAIClient()                       # default: gpt-4o-mini
    llm = OpenAIClient(model="gpt-4o")         # custom model
    llm = OpenAIClient(base_url="http://localhost:11434/v1")  # Ollama

    # Simple interface
    reply = llm.call([{"role": "user", "content": "Hello!"}])

    # Streaming
    for chunk in llm.stream([{"role": "user", "content": "Hello!"}]):
        print(chunk, end="", flush=True)

    # BaseLLM interface
    reply = llm.generate("What is 2+2?")
    response = llm.chat([{"role": "user", "content": "Hello!"}])
"""

from __future__ import annotations

import json
import os
from collections.abc import Generator
from typing import Any

from openai import OpenAI

from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall


class OpenAIClient(BaseLLM):
    """OpenAI-compatible LLM client.

    Reads the API key from the ``OPENAI_API_KEY`` environment variable.
    Raises ``ValueError`` if the key is not set.

    Args:
        model: Model name (default ``"gpt-4o-mini"``).
        api_key: Explicit API key.  If ``None``, reads from environment.
        base_url: Override base URL for compatible endpoints.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> None:

        # Resolve API key: explicit > env var
        resolved_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. "
                "Either pass api_key=... or set the OPENAI_API_KEY environment variable."
            )

        resolved_base = base_url or os.environ.get("OPENAI_BASE_URL")

        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base)
        self.model = model

    # ------------------------------------------------------------------
    # Simple interface
    # ------------------------------------------------------------------

    def call(self, messages: list[dict[str, str]]) -> str:
        """Send messages and return the assistant's reply as a string.

        This is the simplest interface — just text in, text out.

        Args:
            messages: List of message dicts with ``role`` and ``content``.

        Returns:
            The assistant's reply text.
        """
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,
        )
        return completion.choices[0].message.content or ""

    # ------------------------------------------------------------------
    # BaseLLM interface
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        """Send a chat completion request.

        Compatible with the ``BaseLLM`` interface.  Supports function calling
        when *tools* are provided.
        """
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
                tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

        return LLMResponse(
            content=message.content or "",
            tool_calls=tool_calls,
            raw=completion,
        )

    def generate(self, prompt: str) -> str:
        """Send a single prompt string, get text back.

        Compatible with the ``BaseLLM`` interface used by the agent loop.
        """
        return self.call([{"role": "user", "content": prompt}])

    # ------------------------------------------------------------------
    # Streaming interface
    # ------------------------------------------------------------------

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> Generator[str, None, None]:
        """Stream a chat completion response token by token.

        Yields content chunks as they arrive.  Does **not** yield tool call
        fragments — use ``chat()`` for function calling.

        Args:
            messages: OpenAI-format message list.
            tools: Tool definitions (ignored in streaming mode).
            temperature: Sampling temperature.

        Yields:
            Content string fragments.

        Usage::

            for chunk in llm.stream([{"role": "user", "content": "Hello!"}]):
                print(chunk, end="", flush=True)
        """
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
            kwargs["tool_choice"] = "auto"

        stream = self._client.chat.completions.create(**kwargs)
        for event in stream:
            delta = event.choices[0].delta if event.choices else None
            if delta and delta.content:
                yield delta.content

    def call_stream(self, messages: list[dict[str, Any]]) -> Generator[str, None, None]:
        """Simplified streaming interface — yields content chunks.

        Convenience wrapper around ``stream()`` without tool support.
        """
        yield from self.stream(messages)

    def count_tokens(self, messages: list[dict[str, Any]]) -> int:
        """Use tiktoken for accurate token counting when available."""
        try:
            import tiktoken

            enc = tiktoken.encoding_for_model(self.model)
        except Exception:
            return super().count_tokens(messages)

        total = 0
        for msg in messages:
            total += 4  # per-message overhead
            content = msg.get("content") or ""
            total += len(enc.encode(content))
        return total
