"""Tests for the multi-agent prototype."""

import json
from typing import Any

from miniclaw.llm.fake import FakeLLM
from miniclaw.multiagent.agents import (
    PlannerAgent,
    CoderAgent,
    ReviewerAgent,
)
from miniclaw.multiagent.coordinator import Coordinator, PipelineResult
from miniclaw.tools.base import Tool
from miniclaw.tools.registry import ToolRegistry


# ============================================================
# Test fixtures
# ============================================================


class EchoTool(Tool):
    name = "echo"
    description = "Echo."
    schema = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def run(self, text: str, **kwargs: Any) -> str:
        return text


def _make_llm(responses: list[str]) -> FakeLLM:
    return FakeLLM(responses)


def _make_tools() -> ToolRegistry:
    reg = ToolRegistry()
    reg.register(EchoTool())
    return reg


# ============================================================
# Agent roles
# ============================================================


class TestPlannerAgent:
    def test_planner_returns_plan(self):
        llm = _make_llm(
            [
                json.dumps(
                    {
                        "type": "final_answer",
                        "thought": "Analyzing.",
                        "answer": "## Plan\n1. Read files\n2. Summarize",
                    }
                ),
            ]
        )
        agent = PlannerAgent(llm)
        result = agent.run("Analyze the project")
        assert result.success is True
        assert "Plan" in result.answer

    def test_planner_uses_planner_prompt(self):
        llm = _make_llm(
            [
                json.dumps({"type": "final_answer", "thought": "", "answer": "Plan done."}),
            ]
        )
        agent = PlannerAgent(llm)
        agent.run("task")
        # Check that the prompt includes planner-specific text
        prompt = llm.call_log[0]["prompt"]
        assert "Planner" in prompt or "plan" in prompt.lower()


class TestCoderAgent:
    def test_coder_returns_code(self):
        llm = _make_llm(
            [
                json.dumps(
                    {"type": "final_answer", "thought": "Coding.", "answer": "def hello(): pass"}
                ),
            ]
        )
        agent = CoderAgent(llm)
        result = agent.run("Write a hello function")
        assert result.success is True
        assert "hello" in result.answer


class TestReviewerAgent:
    def test_reviewer_returns_review(self):
        llm = _make_llm(
            [
                json.dumps(
                    {
                        "type": "final_answer",
                        "thought": "Reviewing.",
                        "answer": "## Review\nVerdict: APPROVED",
                    }
                ),
            ]
        )
        agent = ReviewerAgent(llm)
        result = agent.run("Review this code")
        assert result.success is True
        assert "APPROVED" in result.answer


# ============================================================
# Coordinator pipeline
# ============================================================


class TestCoordinator:
    def test_full_pipeline(self):
        """Planner → Coder → Reviewer all succeed."""
        llm = _make_llm(
            [
                # Planner
                json.dumps(
                    {
                        "type": "final_answer",
                        "thought": "",
                        "answer": "## Plan\n1. List files\n2. Summarize",
                    }
                ),
                # Coder
                json.dumps(
                    {"type": "final_answer", "thought": "", "answer": "## Code\nprint('hello')"}
                ),
                # Reviewer
                json.dumps(
                    {"type": "final_answer", "thought": "", "answer": "## Review\nAPPROVED"}
                ),
            ]
        )
        coord = Coordinator(llm)
        result = coord.run("Analyze project")

        assert result.success is True
        assert "Plan" in result.plan
        assert "Code" in result.code
        assert "APPROVED" in result.review
        assert result.final_answer == result.review
        assert result.steps > 0

    def test_pipeline_planner_failure(self):
        """Planner fails → pipeline stops."""
        llm = _make_llm(
            [
                # Planner fails (bad JSON)
                "not valid json at all",
                "still bad",
            ]
        )
        coord = Coordinator(llm, _make_tools())
        result = coord.run("task")

        assert result.success is False
        assert "Planner failed" in result.error

    def test_pipeline_coder_failure(self):
        """Coder fails → pipeline stops, plan is preserved."""
        llm = _make_llm(
            [
                # Planner succeeds
                json.dumps({"type": "final_answer", "thought": "", "answer": "Plan."}),
                # Coder fails (3 bad responses → hits max_errors=3)
                "bad json 1",
                "bad json 2",
                "bad json 3",
            ]
        )
        coord = Coordinator(llm, _make_tools())
        result = coord.run("task")

        assert result.success is False
        assert "Coder failed" in result.error
        assert result.plan == "Plan."

    def test_pipeline_reviewer_failure(self):
        """Reviewer fails → pipeline stops, plan and code preserved."""
        llm = _make_llm(
            [
                # Planner
                json.dumps({"type": "final_answer", "thought": "", "answer": "Plan."}),
                # Coder
                json.dumps({"type": "final_answer", "thought": "", "answer": "Code."}),
                # Reviewer fails
                "bad json",
                "still bad",
            ]
        )
        coord = Coordinator(llm, _make_tools())
        result = coord.run("task")

        assert result.success is False
        assert "Reviewer failed" in result.error
        assert result.plan == "Plan."
        assert result.code == "Code."

    def test_pipeline_with_tool_calls(self):
        """Agent uses tools within the pipeline."""
        llm = _make_llm(
            [
                # Planner: just plan
                json.dumps(
                    {"type": "final_answer", "thought": "", "answer": "Plan: use echo tool"}
                ),
                # Coder: call tool, then answer
                json.dumps(
                    {
                        "type": "tool_call",
                        "thought": "Calling echo.",
                        "tool_name": "echo",
                        "arguments": {"text": "hello"},
                    }
                ),
                json.dumps(
                    {"type": "final_answer", "thought": "", "answer": "Code: echo returned hello"}
                ),
                # Reviewer
                json.dumps({"type": "final_answer", "thought": "", "answer": "APPROVED"}),
            ]
        )
        coord = Coordinator(llm, _make_tools())
        result = coord.run("task")

        assert result.success is True
        assert "hello" in result.code

    def test_pipeline_result_repr(self):
        llm = _make_llm(
            [
                json.dumps({"type": "final_answer", "thought": "", "answer": "P"}),
                json.dumps({"type": "final_answer", "thought": "", "answer": "C"}),
                json.dumps({"type": "final_answer", "thought": "", "answer": "R"}),
            ]
        )
        coord = Coordinator(llm)
        result = coord.run("task")
        assert "success" in repr(result)

    def test_pipeline_result_repr_failure(self):
        llm = _make_llm(["bad", "bad"])
        coord = Coordinator(llm, _make_tools())
        result = coord.run("task")
        assert "failed" in repr(result)


# ============================================================
# PipelineResult
# ============================================================


class TestPipelineResult:
    def test_success_result(self):
        r = PipelineResult(success=True, plan="P", code="C", review="R", final_answer="R", steps=5)
        assert r.success is True
        assert r.final_answer == "R"
        assert r.steps == 5

    def test_failure_result(self):
        r = PipelineResult(success=False, error="Planner failed", steps=2)
        assert r.success is False
        assert r.error == "Planner failed"

    def test_defaults(self):
        r = PipelineResult(success=True)
        assert r.plan == ""
        assert r.code == ""
        assert r.review == ""
        assert r.final_answer == ""
        assert r.error is None
        assert r.steps == 0
