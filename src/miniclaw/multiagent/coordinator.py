"""Coordinator — sequential orchestration of Planner → Coder → Reviewer.

The Coordinator runs three agents in sequence, passing each agent's
output as context to the next.  No parallelism, no complex messaging —
just a simple pipeline.

Flow::

    User Task
        │
        ▼
    ┌─────────┐    plan     ┌─────────┐    code     ┌──────────┐
    │ Planner │──────────▶│  Coder  │──────────▶│ Reviewer │
    └─────────┘            └─────────┘            └──────────┘
                                                       │
                                                       ▼
                                                  Final Answer
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from miniclaw.multiagent.agents import (
    CODER_PROMPT,
    PLANNER_PROMPT,
    REVIEWER_PROMPT,
    CoderAgent,
    PlannerAgent,
    ReviewerAgent,
)
from miniclaw.llm.base import BaseLLM
from miniclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


@dataclass
class PipelineResult:
    """Result of the full Planner → Coder → Reviewer pipeline.

    Attributes:
        success: True if all three agents completed successfully.
        plan: The Planner's output.
        code: The Coder's output.
        review: The Reviewer's output.
        final_answer: The last agent's answer (the review).
        error: Error message if the pipeline failed.
        steps: Total steps consumed across all agents.
    """

    success: bool
    plan: str = ""
    code: str = ""
    review: str = ""
    final_answer: str = ""
    error: str | None = None
    steps: int = 0

    def __repr__(self) -> str:
        if self.success:
            return f"<PipelineResult success steps={self.steps}>"
        return f"<PipelineResult failed error={self.error!r}>"


class Coordinator:
    """Sequential coordinator: Planner → Coder → Reviewer.

    Each agent is an ``AgentLoop`` instance with its own system prompt.
    The Coordinator passes the task through all three agents in order,
    feeding each agent's output as context to the next.

    Usage::

        from miniclaw.llm.fake import FakeLLM
        from miniclaw.multiagent import Coordinator

        llm = FakeLLM([...])
        coord = Coordinator(llm)
        result = coord.run("Analyze and improve main.py")
        print(result.final_answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
    ) -> None:
        """Initialize the coordinator with shared LLM and tools.

        Args:
            llm: The LLM backend for all three agents.
            tools: Shared tool registry.  If ``None``, each agent
                gets an empty registry.
        """
        shared_tools = tools or ToolRegistry()
        self.planner = PlannerAgent(llm, shared_tools)
        self.coder = CoderAgent(llm, shared_tools)
        self.reviewer = ReviewerAgent(llm, shared_tools)

    def run(self, task: str) -> PipelineResult:
        """Run the full pipeline on a task.

        Args:
            task: The user's task description.

        Returns:
            A ``PipelineResult`` with outputs from all three agents.
        """
        total_steps = 0

        # --- Step 1: Planner ---
        logger.info("Pipeline: Planner starting")
        planner_input = f"{PLANNER_PROMPT}\n\n---\n\nTask: {task}"
        plan_result = self.planner.run(planner_input)

        total_steps += plan_result.steps_taken
        if not plan_result.success:
            return PipelineResult(
                success=False,
                error=f"Planner failed: {plan_result.error}",
                steps=total_steps,
            )

        plan = plan_result.answer or ""
        logger.info("Pipeline: Planner done (%d steps)", plan_result.steps_taken)

        # --- Step 2: Coder ---
        logger.info("Pipeline: Coder starting")
        coder_input = f"{CODER_PROMPT}\n\n---\n\nTask: {task}\n\n## Plan from Planner\n\n{plan}"
        code_result = self.coder.run(coder_input)

        total_steps += code_result.steps_taken
        if not code_result.success:
            return PipelineResult(
                success=False,
                plan=plan,
                error=f"Coder failed: {code_result.error}",
                steps=total_steps,
            )

        code = code_result.answer or ""
        logger.info("Pipeline: Coder done (%d steps)", code_result.steps_taken)

        # --- Step 3: Reviewer ---
        logger.info("Pipeline: Reviewer starting")
        reviewer_input = (
            f"{REVIEWER_PROMPT}\n\n---\n\n"
            f"Task: {task}\n\n"
            f"## Plan\n\n{plan}\n\n"
            f"## Coder Output\n\n{code}"
        )
        review_result = self.reviewer.run(reviewer_input)

        total_steps += review_result.steps_taken
        if not review_result.success:
            return PipelineResult(
                success=False,
                plan=plan,
                code=code,
                error=f"Reviewer failed: {review_result.error}",
                steps=total_steps,
            )

        review = review_result.answer or ""
        logger.info("Pipeline: Reviewer done (%d steps)", review_result.steps_taken)

        return PipelineResult(
            success=True,
            plan=plan,
            code=code,
            review=review,
            final_answer=review,
            steps=total_steps,
        )
