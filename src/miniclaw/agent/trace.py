"""Structured trace logger for agent loop steps.

Each step in the agent loop produces a ``StepTrace`` that records
everything that happened: the LLM output, parsed action, tool execution,
and any errors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepTrace(BaseModel):
    """A single step's trace record.

    Attributes:
        step: Step number (1-indexed).
        timestamp: Unix timestamp of this step.
        model_output: Raw string from the LLM.
        parsed_action: "tool_call" or "final_answer" or "parse_error".
        tool_name: Name of the tool called (if tool_call).
        arguments: Arguments passed to the tool (if tool_call).
        observation: Tool execution result (if tool_call).
        error: Error message, if any.
    """

    step: int
    timestamp: float = Field(default_factory=time.time)
    model_output: str = ""
    parsed_action: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    observation: Any = None
    error: str | None = None


class TraceLogger:
    """Collects ``StepTrace`` records for an agent run.

    Usage::

        trace = TraceLogger()
        trace.log_step(step=1, model_output='{"type": "tool_call", ...}', ...)
        trace.log_step(step=2, model_output='{"type": "final_answer", ...}', ...)

        # Access traces
        for t in trace.steps:
            print(t.step, t.parsed_action)

        # Export to JSONL
        trace.export_jsonl("run.trace.jsonl")
    """

    def __init__(self) -> None:
        self.steps: list[StepTrace] = []

    def log_step(
        self,
        step: int,
        model_output: str = "",
        parsed_action: str = "",
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: Any = None,
        error: str | None = None,
    ) -> StepTrace:
        """Record a trace for one agent step.

        Returns:
            The created ``StepTrace``.
        """
        trace = StepTrace(
            step=step,
            model_output=model_output,
            parsed_action=parsed_action,
            tool_name=tool_name,
            arguments=arguments,
            observation=observation,
            error=error,
        )
        self.steps.append(trace)
        return trace

    def export_jsonl(self, path: str | Path) -> None:
        """Write all traces to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for trace in self.steps:
                f.write(trace.model_dump_json() + "\n")

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all traces as plain dicts."""
        return [t.model_dump() for t in self.steps]

    def __len__(self) -> int:
        return len(self.steps)
