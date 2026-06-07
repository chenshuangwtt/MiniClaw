"""Agent role definitions for the multi-agent pipeline.

Each role is a thin wrapper around ``AgentLoop`` with a role-specific
system prompt and tool set.  The ``create_*`` factory functions accept
an ``LLM`` and ``ToolRegistry`` so the caller controls which backend
and tools each agent uses.
"""

from __future__ import annotations


from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.base import BaseLLM
from miniclaw.tools.registry import ToolRegistry


# ------------------------------------------------------------------
# System prompts
# ------------------------------------------------------------------

PLANNER_PROMPT = """\
You are the **Planner** agent. Your job is to analyze a task and produce a clear, step-by-step plan.

Rules:
1. Break the task into numbered steps.
2. Identify which tools each step needs.
3. Flag any risks or ambiguities.
4. Do NOT write code — only plan.

Output format:
{
    "type": "final_answer",
    "thought": "your reasoning",
    "answer": "## Plan\\n1. Step one\\n2. Step two\\n..."
}
"""

CODER_PROMPT = """\
You are the **Coder** agent. Your job is to execute a plan by calling tools and producing code or results.

Rules:
1. Follow the plan step by step.
2. Use tools to gather information (read files, list directories, run commands).
3. Write clean, working code.
4. If something fails, adapt and try again.

Output format (when done):
{
    "type": "final_answer",
    "thought": "your reasoning",
    "answer": "the code or results"
}
"""

REVIEWER_PROMPT = """\
You are the **Reviewer** agent. Your job is to review the Coder's output and produce a final assessment.

Rules:
1. Check correctness — does the output match the plan?
2. Check quality — is the code clean, readable, correct?
3. Identify issues and suggest fixes.
4. If the output is good, approve it. If not, explain what's wrong.

Output format:
{
    "type": "final_answer",
    "thought": "your review reasoning",
    "answer": "## Review\\n**Verdict**: APPROVED / NEEDS_FIXES\\n**Details**: ..."
}
"""


# ------------------------------------------------------------------
# Factory functions
# ------------------------------------------------------------------


def create_planner(llm: BaseLLM, tools: ToolRegistry | None = None) -> AgentLoop:
    """Create a Planner agent with read-only tools."""
    return AgentLoop(
        llm=llm,
        registry=tools or ToolRegistry(),
        max_steps=3,
        max_errors=2,
    )


def create_coder(llm: BaseLLM, tools: ToolRegistry | None = None) -> AgentLoop:
    """Create a Coder agent with full tool access."""
    return AgentLoop(
        llm=llm,
        registry=tools or ToolRegistry(),
        max_steps=8,
        max_errors=3,
    )


def create_reviewer(llm: BaseLLM, tools: ToolRegistry | None = None) -> AgentLoop:
    """Create a Reviewer agent with read-only tools."""
    return AgentLoop(
        llm=llm,
        registry=tools or ToolRegistry(),
        max_steps=3,
        max_errors=2,
    )


# ------------------------------------------------------------------
# Convenience classes (thin wrappers)
# ------------------------------------------------------------------


class PlannerAgent:
    """Planner: analyzes tasks and produces structured plans."""

    PROMPT = PLANNER_PROMPT

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None) -> None:
        self.agent = create_planner(llm, tools)

    def run(self, task: str):
        return self.agent.run(f"{PLANNER_PROMPT}\n\n---\n\n{task}")


class CoderAgent:
    """Coder: executes plans using tools and produces code/results."""

    PROMPT = CODER_PROMPT

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None) -> None:
        self.agent = create_coder(llm, tools)

    def run(self, task: str):
        return self.agent.run(f"{CODER_PROMPT}\n\n---\n\n{task}")


class ReviewerAgent:
    """Reviewer: reviews the Coder's output and gives a final assessment."""

    PROMPT = REVIEWER_PROMPT

    def __init__(self, llm: BaseLLM, tools: ToolRegistry | None = None) -> None:
        self.agent = create_reviewer(llm, tools)

    def run(self, task: str):
        return self.agent.run(f"{REVIEWER_PROMPT}\n\n---\n\n{task}")
