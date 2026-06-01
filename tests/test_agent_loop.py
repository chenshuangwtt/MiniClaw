"""Tests for agent_loop.py — integration tests using FakeLLM."""

import pytest
from miniclaw.agent_loop import Agent
from miniclaw.llm.fake import FakeLLM
from miniclaw.tool_registry import ToolRegistry
from miniclaw.trace import TraceLogger


class TestAgentBasic:
    def test_simple_text_response(self):
        llm = FakeLLM(["Hello! I'm here to help."])
        agent = Agent(llm=llm)
        result = agent.run("Hi")
        assert result == "Hello! I'm here to help."

    def test_tool_call_and_response(self):
        """LLM calls a tool, gets result, then gives final answer."""
        llm = FakeLLM([
            '{"tool_call": {"name": "add", "arguments": {"a": 2, "b": 3}}}',
            "The answer is 5.",
        ])
        tools = ToolRegistry()

        @tools.register(name="add", description="Add two numbers.")
        def add(a: int, b: int) -> int:
            return a + b

        agent = Agent(llm=llm, tools=tools, max_turns=5)
        result = agent.run("What is 2+3?")
        assert "5" in result

    def test_unknown_tool_handled(self):
        """Agent should handle calls to unregistered tools gracefully."""
        llm = FakeLLM([
            '{"tool_call": {"name": "nope", "arguments": {}}}',
            "Sorry, I couldn't use that tool.",
        ])
        agent = Agent(llm=llm, max_turns=5)
        result = agent.run("Do something")
        assert "Sorry" in result

    def test_max_turns_exceeded(self):
        """If the LLM keeps calling tools, agent should stop."""
        # 15 tool calls in a row (exceeds default max_turns=10)
        responses = []
        for i in range(15):
            responses.append(f'{{"tool_call": {{"name": "echo", "arguments": {{"text": "{i}"}}}}}}')

        llm = FakeLLM(responses)
        tools = ToolRegistry()

        @tools.register(name="echo", description="Echo.")
        def echo(text: str) -> str:
            return text

        agent = Agent(llm=llm, tools=tools, max_turns=3)
        result = agent.run("go")
        assert "exceeded" in result.lower()

    def test_multiple_tool_calls_in_sequence(self):
        """LLM makes two tool calls before answering."""
        llm = FakeLLM([
            '{"tool_call": {"name": "double", "arguments": {"x": 5}}}',
            '{"tool_call": {"name": "double", "arguments": {"x": 10}}}',
            "I doubled both: 10 and 20.",
        ])
        tools = ToolRegistry()

        @tools.register(name="double", description="Double a number.")
        def double(x: int) -> int:
            return x * 2

        agent = Agent(llm=llm, tools=tools, max_turns=5)
        result = agent.run("Double 5 and 10")
        assert "10" in result
        assert "20" in result

    def test_system_prompt_included(self):
        """Verify the system prompt is sent to the LLM."""
        llm = FakeLLM(["ok"])
        agent = Agent(llm=llm, system_prompt="You are a pirate.")
        agent.run("hello")
        messages = llm.call_log[0]["messages"]
        assert messages[0]["role"] == "system"
        assert "pirate" in messages[0]["content"]

    def test_trace_logging(self):
        """Events are recorded when trace is provided."""
        llm = FakeLLM(["response"])
        trace = TraceLogger(console=False)
        agent = Agent(llm=llm, trace=trace)
        agent.run("test")
        events = trace.get_events()
        assert any(e["type"] == "llm_response" for e in events)


class TestAgentWithMemory:
    def test_memory_loaded_into_messages(self):
        from miniclaw.memory import Memory

        mem = Memory(":memory:")
        mem.append_message("user", "My name is Alice")
        mem.append_message("assistant", "Nice to meet you, Alice!")

        llm = FakeLLM(["I remember you, Alice!"])
        agent = Agent(llm=llm, memory=mem)
        agent.run("Remember me?")

        messages = llm.call_log[0]["messages"]
        # Should have: system + 2 history + user
        assert len(messages) >= 4
        assert any("Alice" in m.get("content", "") for m in messages)
        mem.close()
