"""Prompt templates for the agent loop.

The prompt is assembled in three layers:

1. **System prompt** — role definition and general instructions.
2. **Tools prompt** — JSON Schema descriptions of available tools.
3. **Task prompt** — the user's task plus conversation history.
"""

from __future__ import annotations

import json
from typing import Any

SYSTEM_PROMPT = """\
You are a helpful AI assistant that solves tasks by calling tools.

When you need to call a tool, respond with a JSON object:
{
    "type": "tool_call",
    "thought": "<your reasoning>",
    "tool_name": "<name>",
    "arguments": { ... }
}

When you have enough information to answer, respond with:
{
    "type": "final_answer",
    "thought": "<your reasoning>",
    "answer": "<your final answer>"
}

IMPORTANT: Always respond with exactly ONE JSON object. No extra text.
"""


def build_system_prompt() -> str:
    """Return the base system prompt."""
    return SYSTEM_PROMPT


def build_tools_prompt(tools: list[dict[str, Any]]) -> str:
    """Build a description of available tools from their JSON Schemas.

    Args:
        tools: List of tool schemas, each with ``name``, ``description``,
               and ``parameters`` keys.

    Returns:
        A formatted string listing all tools.
    """
    if not tools:
        return ""

    lines = ["## Available Tools\n"]
    for tool in tools:
        name = tool.get("name", "unknown")
        desc = tool.get("description", "")
        params = tool.get("parameters", {})
        lines.append(f"### {name}")
        lines.append(f"{desc}\n")
        lines.append(
            f"Parameters:\n```json\n{json.dumps(params, indent=2, ensure_ascii=False)}\n```\n"
        )

    return "\n".join(lines)


def build_task_prompt(user_task: str, history: list[dict[str, Any]] | None = None) -> str:
    """Build the task section of the prompt.

    Args:
        user_task: The user's task description.
        history: Optional list of prior steps, each with ``step``,
                 ``action``, ``observation`` keys.

    Returns:
        A formatted string with the task and any history.
    """
    parts = [f"## Task\n\n{user_task}\n"]

    if history:
        parts.append("## Steps So Far\n")
        for entry in history:
            step = entry.get("step", "?")
            action = entry.get("action", "")
            observation = entry.get("observation", "")
            parts.append(f"### Step {step}")
            parts.append(f"Action: {action}")
            parts.append(f"Observation: {observation}\n")

    return "\n".join(parts)


def build_memory_prompt(memories: list[str]) -> str:
    """Build a memory section from a list of memory strings.

    Args:
        memories: List of memory texts from the memory backend.

    Returns:
        A formatted string, or empty string if no memories.
    """
    if not memories:
        return ""

    lines = ["## Long-Term Memory\n"]
    lines.append("The following memories may be relevant:")
    for m in memories:
        lines.append(f"- {m}")
    lines.append("\nUse these memories only when relevant.")
    lines.append("Do not mention them unless useful.")
    return "\n".join(lines)


def build_full_prompt(
    user_task: str,
    tools: list[dict[str, Any]] | None = None,
    history: list[dict[str, Any]] | None = None,
    memories: list[str] | None = None,
) -> str:
    """Assemble the complete prompt from all layers.

    Args:
        user_task: The user's task description.
        tools: List of tool schemas (from ``ToolRegistry.get_schema()``).
        history: Optional list of prior steps.
        memories: Optional list of memory strings from the memory backend.

    Returns:
        The full prompt string ready for ``llm.generate()``.
    """
    sections = [build_system_prompt()]

    if tools:
        sections.append(build_tools_prompt(tools))

    memory_text = build_memory_prompt(memories or [])
    if memory_text:
        sections.append(memory_text)

    sections.append(build_task_prompt(user_task, history))

    return "\n---\n\n".join(sections)
