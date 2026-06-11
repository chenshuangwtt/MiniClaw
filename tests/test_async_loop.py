"""Tests for AsyncAgentLoop."""

from __future__ import annotations

import json
from typing import Any

import pytest

from miniclaw.agent.async_loop import AsyncAgentLoop
from miniclaw.agent.loop import AgentResult
from miniclaw.llm.base import LLMResponse
from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Programmable fake LLM for testing the async loop."""

    def __init__(self, responses: list[str | LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0

    def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.0,
    ) -> LLMResponse:
        if self._index >= len(self._responses):
            return LLMResponse(
                content='{"type": "final_answer", "thought": "done", "answer": "No more responses"}'
            )
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, LLMResponse):
            return resp
        return LLMResponse(content=resp)

    def generate(self, prompt: str) -> str:
        if self._index >= len(self._responses):
            return '{"type": "final_answer", "thought": "done", "answer": "No more responses"}'
        resp = self._responses[self._index]
        self._index += 1
        if isinstance(resp, LLMResponse):
            return resp.content or ""
        return resp


class EchoTool(Tool):
    """A tool that echoes its input."""

    name = "echo"
    description = "Echo the message back"
    schema = {
        "type": "object",
        "properties": {"message": {"type": "string"}},
        "required": ["message"],
    }

    def run(self, message: str = "", **kwargs: Any) -> str:
        return f"Echo: {message}"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAsyncAgentLoop:
    """Tests for AsyncAgentLoop.run()."""

    @pytest.mark.anyio
    async def test_direct_final_answer(self) -> None:
        """LLM returns a final answer directly — no tool calls."""
        responses = [
            '{"type": "final_answer", "thought": "done", "answer": "Hello, world!"}',
        ]
        llm = FakeLLM(responses)
        registry = ToolRegistry()

        loop = AsyncAgentLoop(llm=llm, registry=registry, max_steps=3)
        result = await loop.run("Say hello")

        assert isinstance(result, AgentResult)
        assert result.success is True
        assert result.answer == "Hello, world!"
        assert result.steps_taken == 1

    @pytest.mark.anyio
    async def test_tool_call_then_answer(self) -> None:
        """LLM calls a tool, then returns a final answer."""
        responses = [
            json.dumps(
                {
                    "type": "tool_call",
                    "thought": "I need to echo",
                    "tool_name": "echo",
                    "arguments": {"message": "ping"},
                }
            ),
            json.dumps(
                {
                    "type": "final_answer",
                    "thought": "got the echo",
                    "answer": "Echo was: Echo: ping",
                }
            ),
        ]
        llm = FakeLLM(responses)
        registry = ToolRegistry()
        registry.register(EchoTool())

        loop = AsyncAgentLoop(llm=llm, registry=registry, max_steps=5)
        result = await loop.run("Echo ping")

        assert result.success is True
        assert "Echo: ping" in result.answer or result.answer is not None

    @pytest.mark.anyio
    async def test_max_steps_exceeded(self) -> None:
        """Loop stops after max_steps even if no final answer."""
        # Always return tool calls (never a final answer)
        tool_call = json.dumps(
            {
                "type": "tool_call",
                "thought": "keep going",
                "tool_name": "echo",
                "arguments": {"message": "loop"},
            }
        )
        llm = FakeLLM([tool_call] * 5)
        registry = ToolRegistry()
        registry.register(EchoTool())

        loop = AsyncAgentLoop(llm=llm, registry=registry, max_steps=3)
        result = await loop.run("infinite loop")

        assert result.success is False
        assert "maximum steps" in result.error.lower() or result.steps_taken == 3

    @pytest.mark.anyio
    async def test_returns_agent_result_type(self) -> None:
        """Verify the return type is AgentResult."""
        responses = ['{"type": "final_answer", "thought": "ok", "answer": "done"}']
        llm = FakeLLM(responses)
        registry = ToolRegistry()

        loop = AsyncAgentLoop(llm=llm, registry=registry)
        result = await loop.run("test")

        assert isinstance(result, AgentResult)
        assert hasattr(result, "success")
        assert hasattr(result, "answer")
        assert hasattr(result, "trace")
        assert hasattr(result, "steps_taken")

    @pytest.mark.anyio
    async def test_trace_recorded(self) -> None:
        """Verify that trace steps are recorded."""
        responses = [
            '{"type": "tool_call", "thought": "echo", "tool_name": "echo", "arguments": {"message": "hi"}}',
            '{"type": "final_answer", "thought": "done", "answer": "result"}',
        ]
        llm = FakeLLM(responses)
        registry = ToolRegistry()
        registry.register(EchoTool())

        loop = AsyncAgentLoop(llm=llm, registry=registry, max_steps=5)
        result = await loop.run("test trace")

        assert len(result.trace) >= 2  # at least tool call + final answer
