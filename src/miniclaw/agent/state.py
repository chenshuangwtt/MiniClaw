"""Pydantic models for structured agent output.

The LLM is expected to return one of two JSON shapes:

Tool call::

    {
        "type": "tool_call",
        "thought": "I need to look up the weather.",
        "tool_name": "get_weather",
        "arguments": {"city": "Beijing"}
    }

Final answer::

    {
        "type": "final_answer",
        "thought": "I now have enough information.",
        "answer": "The weather in Beijing is sunny, 25°C."
    }
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal, Union

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    """The LLM wants to invoke a tool."""

    type: Literal["tool_call"] = "tool_call"
    thought: str = Field(
        default="", description="The LLM's chain-of-thought before deciding to call a tool."
    )
    tool_name: str = Field(description="Name of the tool to invoke.")
    arguments: dict[str, Any] = Field(
        default_factory=dict, description="Arguments to pass to the tool."
    )
    tool_call_id: str | None = Field(
        default=None, description="Provider-generated ID for native tool calls."
    )


class FinalAnswer(BaseModel):
    """The LLM has produced its final answer — no more tool calls."""

    type: Literal["final_answer"] = "final_answer"
    thought: str = Field(
        default="", description="The LLM's chain-of-thought before producing the answer."
    )
    answer: str = Field(description="The final answer text.")


# The union type that the parser returns.
AgentOutput = Union[ToolCall, FinalAnswer]


class RunStatus(StrEnum):
    """Lifecycle state for one agent run."""

    CREATED = "created"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
