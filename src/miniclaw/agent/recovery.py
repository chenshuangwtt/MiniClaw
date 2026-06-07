"""RecoveryManager — graceful error handling for the agent loop.

Every public method returns a **recovery message** (a dict or string) that
is designed to be injected back into the agent's conversation history as an
``observation``.  The message is written in a way that helps the LLM
self-correct on the next iteration.

Design principles:
    1. **Never raise** — every method returns a usable value.
    2. **Be helpful** — include enough context for the LLM to fix itself.
    3. **Be structured** — return dicts so the caller can decide how to use them.
"""

from __future__ import annotations

import json
from typing import Any

from miniclaw.tools.registry import ToolRegistry


class RecoveryManager:
    """Provides recovery strategies for common agent-loop failures.

    Usage::

        recovery = RecoveryManager(max_errors=3)

        # 1. Invalid JSON
        fixed = recovery.handle_invalid_json(raw_output)

        # 2. Unknown tool
        hint = recovery.handle_unknown_tool("ghost", registry)

        # 3. Bad arguments
        hint = recovery.handle_bad_arguments("add", "missing 'a'", registry)

        # 4. Tool execution error
        hint = recovery.handle_tool_error("fail", "RuntimeError: kaboom")

        # 5. Consecutive failures
        abort = recovery.handle_consecutive_failures(error_count=3)
        # → returns dict or None
    """

    def __init__(self, max_errors: int = 3) -> None:
        self.max_errors = max_errors

    # ------------------------------------------------------------------
    # 1. Invalid JSON recovery
    # ------------------------------------------------------------------

    def handle_invalid_json(self, raw_output: str) -> dict[str, str]:
        """Try to repair malformed JSON from the LLM.

        Attempts to extract the first ``{...}`` block from *raw_output*.
        If successful, returns the repaired JSON string.  Otherwise returns
        an error observation that guides the LLM to retry.

        Returns:
            A dict with either ``status="repaired"`` and ``output`` (the
            fixed JSON string), or ``status="failed"`` and ``error``.
        """
        extracted = self._extract_first_json(raw_output)
        if extracted is not None:
            return {"status": "repaired", "output": extracted}

        return {
            "status": "failed",
            "error": (
                "Your response was not valid JSON. "
                "Please respond with exactly ONE JSON object and nothing else. "
                "Required format:\n"
                '  {"type": "tool_call", "thought": "...", "tool_name": "...", "arguments": {...}}\n'
                "or:\n"
                '  {"type": "final_answer", "thought": "...", "answer": "..."}'
            ),
        }

    # ------------------------------------------------------------------
    # 2. Unknown tool recovery
    # ------------------------------------------------------------------

    def handle_unknown_tool(self, tool_name: str, registry: ToolRegistry) -> dict[str, str]:
        """Build a recovery message for an unrecognised tool name.

        Returns:
            A dict with ``status="error"`` listing available tools.
        """
        available = registry.list()
        available_str = ", ".join(available) if available else "(no tools registered)"

        return {
            "status": "error",
            "error": (
                f"Tool '{tool_name}' does not exist. "
                f"Available tools: {available_str}. "
                "Please use one of the available tools, or provide a final_answer."
            ),
        }

    # ------------------------------------------------------------------
    # 3. Bad arguments recovery
    # ------------------------------------------------------------------

    def handle_bad_arguments(
        self,
        tool_name: str,
        error: str,
        registry: ToolRegistry,
    ) -> dict[str, str]:
        """Build a recovery message for invalid tool arguments.

        Includes the tool's JSON Schema so the LLM can correct its call.

        Returns:
            A dict with ``status="error"``, the ``error`` message,
            and the tool's ``schema``.
        """
        schema = registry.get_schema(tool_name)
        schema_str = (
            json.dumps(schema, indent=2, ensure_ascii=False) if schema else "(schema unavailable)"
        )

        return {
            "status": "error",
            "error": (
                f"Invalid arguments for tool '{tool_name}': {error}\n"
                f"Tool schema:\n{schema_str}\n"
                "Please correct the arguments and try again."
            ),
        }

    # ------------------------------------------------------------------
    # 4. Tool execution failure recovery
    # ------------------------------------------------------------------

    def handle_tool_error(self, tool_name: str, error: str) -> dict[str, str]:
        """Format a tool execution error as a model-friendly observation.

        Returns:
            A dict with ``status="error"`` and a descriptive message.
        """
        return {
            "status": "error",
            "error": (
                f"Tool '{tool_name}' failed during execution: {error}\n"
                "You may retry with different arguments, try a different tool, "
                "or provide a final_answer explaining the issue."
            ),
        }

    # ------------------------------------------------------------------
    # 5. Consecutive failure abort
    # ------------------------------------------------------------------

    def handle_consecutive_failures(self, error_count: int) -> dict[str, Any] | None:
        """Check whether the agent should abort due to too many errors.

        Returns:
            A dict with ``status="abort"`` and a final_answer-shaped error
            message if *error_count* >= ``self.max_errors``, otherwise ``None``.
        """
        if error_count < self.max_errors:
            return None

        return {
            "status": "abort",
            "error": (
                f"Aborting after {error_count} consecutive errors. "
                "Unable to complete the task. "
                "Please try rephrasing your request or simplifying the task."
            ),
            # Shaped like a final_answer so the caller can emit it directly
            "type": "final_answer",
            "answer": (
                f"I was unable to complete the task after {error_count} consecutive errors. "
                "Please try rephrasing your request or breaking it into smaller steps."
            ),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_first_json(text: str) -> str | None:
        """Attempt to extract the first ``{...}`` JSON object from *text*.

        Uses brace counting to handle nested objects correctly.

        Returns:
            The JSON string if a valid object was found, otherwise ``None``.
        """
        # Find the first '{'
        start = text.find("{")
        if start == -1:
            return None

        # Count braces to find the matching '}'
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    # Verify it's valid JSON
                    try:
                        json.loads(candidate)
                        return candidate
                    except json.JSONDecodeError:
                        return None

        return None
