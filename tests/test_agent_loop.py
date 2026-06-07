"""Tests for the Agent Loop — using FakeLLM, no real API calls."""

import json
from typing import Any
from pathlib import Path

from miniclaw.agent.loop import AgentLoop
from miniclaw.agent.trace import TraceLogger, StepTrace
from miniclaw.agent.prompts import build_full_prompt, build_tools_prompt, build_system_prompt
from miniclaw.llm.fake import FakeLLM
from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


# ============================================================
# Test fixtures
# ============================================================


class EchoTool(Tool):
    name = "echo"
    description = "Echo back the input text."
    schema = {
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    }

    def run(self, text: str, **kwargs: Any) -> str:
        return text


class AddTool(Tool):
    name = "add"
    description = "Add two numbers."
    schema = {
        "type": "object",
        "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
        "required": ["a", "b"],
    }

    def run(self, a: float, b: float, **kwargs: Any) -> float:
        return a + b


class FailingTool(Tool):
    name = "fail"
    description = "Always fails."
    schema = {"type": "object", "properties": {}, "required": []}

    def run(self, **kwargs: Any) -> None:
        raise RuntimeError("intentional failure")


def _make_registry(*tools: Tool) -> ToolRegistry:
    reg = ToolRegistry()
    for t in tools:
        reg.register(t)
    return reg


# ============================================================
# Prompt building tests
# ============================================================


class TestPrompts:
    def test_system_prompt_not_empty(self):
        sp = build_system_prompt()
        assert "JSON" in sp
        assert "tool_call" in sp

    def test_tools_prompt_lists_tools(self):
        schemas = [
            {"name": "echo", "description": "Echo.", "parameters": {"type": "object"}},
            {"name": "add", "description": "Add.", "parameters": {"type": "object"}},
        ]
        prompt = build_tools_prompt(schemas)
        assert "echo" in prompt
        assert "add" in prompt

    def test_tools_prompt_empty(self):
        assert build_tools_prompt([]) == ""

    def test_full_prompt_includes_task(self):
        prompt = build_full_prompt("What is 2+3?", tools=[], history=[])
        assert "What is 2+3?" in prompt

    def test_full_prompt_includes_history(self):
        history = [{"step": 1, "action": "Called add", "observation": "5"}]
        prompt = build_full_prompt("task", tools=[], history=history)
        assert "Called add" in prompt
        assert "5" in prompt


# ============================================================
# TraceLogger tests
# ============================================================


class TestTraceLogger:
    def test_log_step(self):
        trace = TraceLogger()
        t = trace.log_step(step=1, model_output="{}", parsed_action="final_answer")
        assert isinstance(t, StepTrace)
        assert t.step == 1
        assert len(trace) == 1

    def test_log_multiple(self):
        trace = TraceLogger()
        trace.log_step(step=1, model_output="{}", parsed_action="tool_call", tool_name="echo")
        trace.log_step(step=2, model_output="{}", parsed_action="final_answer")
        assert len(trace) == 2

    def test_to_dicts(self):
        trace = TraceLogger()
        trace.log_step(step=1, model_output="raw", parsed_action="tool_call", tool_name="x")
        dicts = trace.to_dicts()
        assert len(dicts) == 1
        assert dicts[0]["step"] == 1
        assert dicts[0]["tool_name"] == "x"

    def test_export_jsonl(self, tmp_path: Path):
        trace = TraceLogger()
        trace.log_step(step=1, parsed_action="a")
        trace.log_step(step=2, parsed_action="b")
        f = tmp_path / "trace.jsonl"
        trace.export_jsonl(f)
        lines = f.read_text().strip().split("\n")
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert obj["step"] == 1

    def test_timestamp_set(self):
        trace = TraceLogger()
        t = trace.log_step(step=1)
        assert t.timestamp > 0


# ============================================================
# AgentLoop — happy path
# ============================================================


class TestAgentLoopBasic:
    def test_immediate_final_answer(self):
        """LLM returns final_answer on the first step."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "I know.", "answer": "42"}',
            ]
        )
        reg = _make_registry(EchoTool())
        agent = AgentLoop(llm=llm, registry=reg)

        result = agent.run("What is the meaning of life?")

        assert result.success is True
        assert result.answer == "42"
        assert result.steps_taken == 1
        assert len(result.trace) == 1

    def test_tool_call_then_final_answer(self):
        """LLM calls a tool, gets result, then gives final answer."""
        llm = FakeLLM(
            [
                '{"type": "tool_call", "thought": "Need to echo.", "tool_name": "echo", "arguments": {"text": "hello"}}',
                '{"type": "final_answer", "thought": "Got it.", "answer": "Result: hello"}',
            ]
        )
        reg = _make_registry(EchoTool())
        agent = AgentLoop(llm=llm, registry=reg)

        result = agent.run("Echo hello")

        assert result.success is True
        assert "hello" in result.answer
        assert result.steps_taken == 2

    def test_multiple_tool_calls(self):
        """LLM calls tools across multiple steps before answering."""
        llm = FakeLLM(
            [
                '{"type": "tool_call", "tool_name": "add", "arguments": {"a": 1, "b": 2}}',
                '{"type": "tool_call", "tool_name": "add", "arguments": {"a": 3, "b": 4}}',
                '{"type": "final_answer", "answer": "1+2=3, 3+4=7"}',
            ]
        )
        reg = _make_registry(AddTool())
        agent = AgentLoop(llm=llm, registry=reg)

        result = agent.run("Add numbers")

        assert result.success is True
        assert result.steps_taken == 3
        # Check trace recorded tool calls
        tool_steps = [t for t in result.trace.steps if t.parsed_action == "tool_call"]
        assert len(tool_steps) == 2

    def test_result_repr(self):
        llm = FakeLLM(['{"type": "final_answer", "answer": "ok"}'])
        result = AgentLoop(llm=llm, registry=_make_registry()).run("x")
        assert "success" in repr(result)


# ============================================================
# AgentLoop — error handling
# ============================================================


class TestAgentLoopErrors:
    def test_tool_error_adds_to_history(self):
        """Tool failure is recorded in history and trace, not a crash."""
        llm = FakeLLM(
            [
                '{"type": "tool_call", "tool_name": "fail", "arguments": {}}',
                '{"type": "final_answer", "answer": "Handled the error."}',
            ]
        )
        reg = _make_registry(FailingTool())
        agent = AgentLoop(llm=llm, registry=reg, max_errors=3)

        result = agent.run("Try failing")

        assert result.success is True
        # The tool error should be in the trace
        error_step = [t for t in result.trace.steps if t.error is not None]
        assert len(error_step) == 1
        assert "intentional" in error_step[0].error

    def test_unknown_tool_recorded(self):
        """Calling a non-existent tool is handled gracefully."""
        llm = FakeLLM(
            [
                '{"type": "tool_call", "tool_name": "ghost", "arguments": {}}',
                '{"type": "final_answer", "answer": "Learned ghost does not exist."}',
            ]
        )
        reg = _make_registry(EchoTool())
        agent = AgentLoop(llm=llm, registry=reg, max_errors=3)

        result = agent.run("Try ghost")

        assert result.success is True
        error_step = [t for t in result.trace.steps if t.error is not None]
        assert len(error_step) == 1
        assert "ghost" in error_step[0].error.lower()
        assert "does not exist" in error_step[0].error.lower()

    def test_max_steps_exceeded(self):
        """Agent stops after max_steps."""
        responses = [
            f'{{"type": "tool_call", "tool_name": "echo", "arguments": {{"text": "{i}"}}}}'
            for i in range(5)
        ]
        llm = FakeLLM(responses)
        reg = _make_registry(EchoTool())
        agent = AgentLoop(llm=llm, registry=reg, max_steps=3)

        result = agent.run("Loop forever")

        assert result.success is False
        assert "Exceeded" in result.error
        assert result.steps_taken == 3

    def test_max_errors_aborts(self):
        """Agent stops after max_errors consecutive failures."""
        # 10 failing tool calls — more than max_errors + max_steps
        responses = [
            '{"type": "tool_call", "tool_name": "fail", "arguments": {}}' for _ in range(10)
        ]
        llm = FakeLLM(responses)
        reg = _make_registry(FailingTool())
        agent = AgentLoop(llm=llm, registry=reg, max_steps=10, max_errors=2)

        result = agent.run("Keep failing")

        assert result.success is False
        assert "error" in result.error.lower()

    def test_parse_error_recorded(self):
        """Invalid LLM output is caught and recorded."""
        llm = FakeLLM(
            [
                "this is not valid JSON at all",
                '{"type": "final_answer", "answer": "Recovered."}',
            ]
        )
        reg = _make_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_errors=3)

        result = agent.run("Try bad output")

        assert result.success is True
        parse_errors = [t for t in result.trace.steps if t.parsed_action == "parse_error"]
        assert len(parse_errors) == 1

    def test_consecutive_parse_errors_abort(self):
        """Too many parse errors cause abort."""
        llm = FakeLLM(
            [
                "bad1",
                "bad2",
                "bad3",
            ]
        )
        reg = _make_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_errors=2)

        result = agent.run("task")

        assert result.success is False
        assert "error" in result.error.lower()

    def test_llm_exception_recorded(self):
        """LLM raising an exception is handled gracefully."""

        class ErrorLLM(FakeLLM):
            def generate(self, prompt: str) -> str:
                raise ConnectionError("network down")

        llm = ErrorLLM(["unused"])
        reg = _make_registry()
        agent = AgentLoop(llm=llm, registry=reg, max_errors=2)

        result = agent.run("task")

        assert result.success is False
        assert "error" in result.error.lower()


# ============================================================
# AgentLoop — trace verification
# ============================================================


class TestAgentLoopTrace:
    def test_trace_records_all_steps(self):
        llm = FakeLLM(
            [
                '{"type": "tool_call", "tool_name": "echo", "arguments": {"text": "hi"}}',
                '{"type": "final_answer", "answer": "done"}',
            ]
        )
        reg = _make_registry(EchoTool())
        agent = AgentLoop(llm=llm, registry=reg)

        result = agent.run("task")

        assert len(result.trace) == 2
        t1 = result.trace.steps[0]
        assert t1.parsed_action == "tool_call"
        assert t1.tool_name == "echo"
        assert t1.arguments == {"text": "hi"}
        assert t1.observation == "hi"
        assert t1.error is None

        t2 = result.trace.steps[1]
        assert t2.parsed_action == "final_answer"
        assert t2.observation == "done"

    def test_trace_tool_error_recorded(self):
        llm = FakeLLM(
            [
                '{"type": "tool_call", "tool_name": "fail", "arguments": {}}',
                '{"type": "final_answer", "answer": "ok"}',
            ]
        )
        reg = _make_registry(FailingTool())
        agent = AgentLoop(llm=llm, registry=reg, max_errors=3)

        result = agent.run("task")

        t1 = result.trace.steps[0]
        assert t1.parsed_action == "tool_call"
        assert t1.tool_name == "fail"
        assert t1.error is not None
        assert "intentional" in t1.error
        assert t1.observation is None

    def test_trace_model_output_saved(self):
        raw = '{"type": "final_answer", "answer": "hello"}'
        llm = FakeLLM([raw])
        result = AgentLoop(llm=llm, registry=_make_registry()).run("task")
        assert result.trace.steps[0].model_output == raw


# ============================================================
# FakeLLM — generate() interface
# ============================================================


class TestFakeLLMGenerate:
    def test_sequential_generate(self):
        llm = FakeLLM(["first", "second", "third"])
        assert llm.generate("p1") == "first"
        assert llm.generate("p2") == "second"
        assert llm.generate("p3") == "third"

    def test_exhausted_returns_default(self):
        llm = FakeLLM(["only"])
        llm.generate("p")
        result = llm.generate("p")
        assert "final_answer" in result
        assert "No more" in result

    def test_reset(self):
        llm = FakeLLM(["a", "b"])
        llm.generate("p")
        llm.generate("p")
        llm.reset()
        assert llm.generate("p") == "a"

    def test_call_log(self):
        llm = FakeLLM(["r1", "r2"])
        llm.generate("prompt1")
        llm.generate("prompt2")
        assert len(llm.call_log) == 2
        assert llm.call_log[0]["prompt"] == "prompt1"


# ============================================================
# Memory integration
# ============================================================


class FakeMemoryBackend:
    """In-memory backend for testing."""

    def __init__(self):
        self.entries: list[dict] = []

    def add(self, text: str, user_id: str, metadata=None):
        self.entries.append({"text": text, "user_id": user_id, "metadata": metadata})

    def search(self, query: str, user_id: str, limit: int = 5) -> list[str]:
        return [e["text"] for e in self.entries if e["user_id"] == user_id][:limit]


class TestAgentLoopMemory:
    def test_memories_injected_into_prompt(self):
        """Memories from backend should appear in the prompt."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()
        mem.add("用户喜欢简洁风格", user_id="alice")

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("分析代码", user_id="alice")

        # Check that the prompt sent to LLM contains the memory
        prompt = llm.call_log[0]["prompt"]
        assert "简洁风格" in prompt
        assert "Long-Term Memory" in prompt

    def test_memories_not_injected_for_different_user(self):
        """Memories for other users should not appear."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()
        mem.add("用户喜欢简洁风格", user_id="alice")

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("分析代码", user_id="bob")

        prompt = llm.call_log[0]["prompt"]
        assert "简洁风格" not in prompt

    def test_no_memories_section_when_empty(self):
        """When no memories exist, the prompt should not have a memory section."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("分析代码", user_id="alice")

        prompt = llm.call_log[0]["prompt"]
        assert "Long-Term Memory" not in prompt

    def test_task_with_keyword_saves_memory(self):
        """Tasks containing memory keywords should be saved."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "好的，已记住。"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("请记住我喜欢用 Python", user_id="alice")

        assert len(mem.entries) == 1
        assert "记住" in mem.entries[0]["text"]
        assert mem.entries[0]["user_id"] == "alice"

    def test_task_without_keyword_no_save(self):
        """Tasks without memory keywords should not be saved."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "好的。"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("分析代码结构", user_id="alice")

        assert len(mem.entries) == 0

    def test_memory_search_failure_does_not_crash(self):
        """Memory search failure should not interrupt the agent."""

        class BrokenMemory:
            def add(self, text, user_id, metadata=None):
                raise RuntimeError("db down")

            def search(self, query, user_id, limit=5):
                raise RuntimeError("db down")

        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=BrokenMemory())
        result = agent.run("请记住这个", user_id="alice")

        # Should still succeed despite memory failure
        assert result.success is True

    def test_default_memory_backend_is_null(self):
        """Without explicit memory_backend, NullMemoryBackend is used."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()

        agent = AgentLoop(llm=llm, registry=reg)
        result = agent.run("请记住我喜欢咖啡", user_id="alice")

        assert result.success is True
        # NullMemoryBackend.search returns [], NullMemoryBackend.add does nothing

    def test_user_id_passed_to_backend(self):
        """user_id should be forwarded to memory backend."""
        llm = FakeLLM(
            [
                '{"type": "final_answer", "thought": "", "answer": "done"}',
            ]
        )
        reg = _make_registry()
        mem = FakeMemoryBackend()

        agent = AgentLoop(llm=llm, registry=reg, memory_backend=mem)
        agent.run("请记住我喜欢咖啡", user_id="bob")

        if mem.entries:
            assert mem.entries[0]["user_id"] == "bob"
