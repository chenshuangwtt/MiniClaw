"""Coordinator — sequential orchestration with conditional retry.

Flow::

    User Task
        │
        ▼
    ┌─────────┐    plan     ┌─────────┐    code     ┌──────────┐
    │ Planner │──────────▶│  Coder  │──────────▶│ Reviewer │
    └─────────┘            └─────────┘            └──────────┘
                                ▲                       │
                                │   NEEDS_FIXES         │
                                └───────────────────────┘
                                                       │
                                                  APPROVED
                                                       │
                                                       ▼
                                                  Final Answer
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

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
    """Result of the Planner → Coder → Reviewer pipeline.

    Attributes:
        success: True if the reviewer approved.
        plan: The Planner's output.
        code: The Coder's final output.
        review: The Reviewer's final output.
        final_answer: The review text.
        error: Error message if the pipeline failed.
        steps: Total steps consumed across all agents.
        attempts: How many Coder→Reviewer cycles were attempted.
        feedback: List of reviewer feedback from each attempt.
    """

    success: bool
    plan: str = ""
    code: str = ""
    review: str = ""
    final_answer: str = ""
    error: str | None = None
    steps: int = 0
    attempts: int = 0
    feedback: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.success:
            return f"<PipelineResult success steps={self.steps} attempts={self.attempts}>"
        return f"<PipelineResult failed error={self.error!r} steps={self.steps}>"


class Coordinator:
    """Sequential coordinator with conditional retry.

    Runs Planner → Coder → Reviewer.  If the Reviewer returns
    ``NEEDS_FIXES``, the Coder re-runs with the reviewer's feedback
    injected into the context.  Retries up to ``max_retries`` times.

    Usage::

        from miniclaw.llm.fake import FakeLLM
        from miniclaw.multiagent import Coordinator

        llm = FakeLLM([...])
        coord = Coordinator(llm, max_retries=2)
        result = coord.run("Analyze and improve main.py")
        print(result.final_answer)
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
        max_retries: int = 2,
    ) -> None:
        """Initialize the coordinator.

        Args:
            llm: The LLM backend for all three agents.
            tools: Shared tool registry.
            max_retries: Maximum Coder→Reviewer retry cycles.
        """
        shared_tools = tools or ToolRegistry()
        self.planner = PlannerAgent(llm, shared_tools)
        self.coder = CoderAgent(llm, shared_tools)
        self.reviewer = ReviewerAgent(llm, shared_tools)
        self.max_retries = max_retries

    def run(self, task: str) -> PipelineResult:
        """Run the full pipeline on a task.

        Args:
            task: The user's task description.

        Returns:
            A ``PipelineResult`` with outputs from all agents.
        """
        total_steps = 0
        feedback_history: list[str] = []

        # --- Step 1: Planner ---
        logger.info("Pipeline: Planner starting")
        plan_result = self.planner.run(f"{PLANNER_PROMPT}\n\n---\n\nTask: {task}")
        total_steps += plan_result.steps_taken

        if not plan_result.success:
            return PipelineResult(
                success=False,
                error=f"Planner failed: {plan_result.error}",
                steps=total_steps,
            )

        plan = plan_result.answer or ""
        logger.info("Pipeline: Planner done (%d steps)", plan_result.steps_taken)

        # --- Step 2+3: Coder → Reviewer loop ---
        code = ""
        review = ""
        for attempt in range(1, self.max_retries + 2):  # +2: first attempt + max_retries retries
            # --- Coder ---
            logger.info("Pipeline: Coder attempt %d", attempt)
            coder_input = self._build_coder_input(task, plan, code, feedback_history)
            code_result = self.coder.run(coder_input)
            total_steps += code_result.steps_taken

            if not code_result.success:
                return PipelineResult(
                    success=False,
                    plan=plan,
                    error=f"Coder failed: {code_result.error}",
                    steps=total_steps,
                    attempts=attempt,
                    feedback=feedback_history,
                )

            code = code_result.answer or ""
            logger.info("Pipeline: Coder done (%d steps)", code_result.steps_taken)

            # --- Reviewer ---
            logger.info("Pipeline: Reviewer attempt %d", attempt)
            reviewer_input = self._build_reviewer_input(task, plan, code)
            review_result = self.reviewer.run(reviewer_input)
            total_steps += review_result.steps_taken

            if not review_result.success:
                return PipelineResult(
                    success=False,
                    plan=plan,
                    code=code,
                    error=f"Reviewer failed: {review_result.error}",
                    steps=total_steps,
                    attempts=attempt,
                    feedback=feedback_history,
                )

            review = review_result.answer or ""
            logger.info("Pipeline: Reviewer done (%d steps)", review_result.steps_taken)

            # --- Check verdict ---
            if _is_approved(review):
                logger.info("Pipeline: APPROVED after %d attempt(s)", attempt)
                return PipelineResult(
                    success=True,
                    plan=plan,
                    code=code,
                    review=review,
                    final_answer=review,
                    steps=total_steps,
                    attempts=attempt,
                    feedback=feedback_history,
                )

            # NEEDS_FIXES — record feedback and retry
            feedback_history.append(review)
            logger.info("Pipeline: NEEDS_FIXES, retrying (%d/%d)", attempt, self.max_retries)

        # Max retries exceeded
        return PipelineResult(
            success=False,
            plan=plan,
            code=code,
            review=review,
            error=f"Reviewer did not approve after {self.max_retries + 1} attempts.",
            steps=total_steps,
            attempts=self.max_retries + 1,
            feedback=feedback_history,
        )

    # ------------------------------------------------------------------
    # Input builders
    # ------------------------------------------------------------------

    def _build_coder_input(
        self,
        task: str,
        plan: str,
        previous_code: str,
        feedback: list[str],
    ) -> str:
        """Build the Coder's input with plan + any reviewer feedback."""
        parts = [CODER_PROMPT, "\n---\n"]
        parts.append(f"Task: {task}")
        parts.append(f"\n## Plan\n\n{plan}")

        if previous_code and feedback:
            parts.append("\n## Previous Attempt\n")
            parts.append(previous_code)
            parts.append(f"\n## Reviewer Feedback\n\n{feedback[-1]}")
            parts.append("\nPlease fix the issues mentioned above.")

        return "\n".join(parts)

    def _build_reviewer_input(self, task: str, plan: str, code: str) -> str:
        """Build the Reviewer's input with task + plan + code."""
        return (
            f"{REVIEWER_PROMPT}\n\n---\n\n"
            f"Task: {task}\n\n"
            f"## Plan\n\n{plan}\n\n"
            f"## Coder Output\n\n{code}"
        )


def _is_approved(review: str) -> bool:
    """Check if the reviewer's verdict is APPROVED."""
    verdict = review.strip()
    if re.search(r"\bNOT\s+APPROVED\b|\bNEEDS[_\s-]?FIXES\b", verdict, re.IGNORECASE):
        return False
    return bool(
        re.search(
            r"(^|\n)\s*(?:verdict\s*:\s*)?APPROVED\b",
            verdict,
            re.IGNORECASE,
        )
    )
