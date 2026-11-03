"""Regression coverage for native provider tool-calling semantics."""

from __future__ import annotations

from typing import Any

from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.base import BaseLLM, LLMResponse, ToolCall
from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


class NativeScriptedLLM(BaseLLM):
    def __init__(self, responses: list[LLMResponse]) -> None:
        self.responses = responses
        self.calls: list[tuple[list[dict[str, Any]], list[dict[str, Any]] | None]] = []

    def chat(self, messages, tools=None, temperature=0.0):
        self.calls.append((messages, tools))
        return self.responses.pop(0)


class Echo(Tool):
    name = "echo"
    description = "Return supplied text."
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
        "additionalProperties": False,
    }

    def run(self, text: str) -> str:
        return text


def test_native_mode_executes_all_calls_and_preserves_ids() -> None:
    llm = NativeScriptedLLM(
        [
            LLMResponse(
                tool_calls=[
                    ToolCall(id="call_a", name="echo", arguments={"text": "one"}),
                    ToolCall(id="call_b", name="echo", arguments={"text": "two"}),
                ]
            ),
            LLMResponse(content="Done."),
        ]
    )
    registry = ToolRegistry()
    registry.register(Echo())
    agent = AgentLoop(llm=llm, registry=registry)
    agent.parse_mode = "native"

    result = agent.run("Echo both values")

    assert result.success is True
    assert result.answer == "Done."
    follow_up_messages = llm.calls[1][0]
    tool_messages = [message for message in follow_up_messages if message["role"] == "tool"]
    assert [message["tool_call_id"] for message in tool_messages] == ["call_a", "call_b"]
    assistant = next(message for message in follow_up_messages if message.get("tool_calls"))
    assert [call["id"] for call in assistant["tool_calls"]] == ["call_a", "call_b"]


def test_native_mode_rejects_invalid_arguments_before_tool_execution() -> None:
    llm = NativeScriptedLLM(
        [
            LLMResponse(
                tool_calls=[ToolCall(id="call_bad", name="echo", arguments={"wrong": "x"})]
            ),
            LLMResponse(content="Cannot finish."),
        ]
    )
    registry = ToolRegistry()
    registry.register(Echo())
    agent = AgentLoop(llm=llm, registry=registry)
    agent.parse_mode = "native"

    result = agent.run("Call echo")

    assert result.success is False
    assert result.trace.steps[0].error is not None
    assert "Missing required argument" in result.trace.steps[0].error
