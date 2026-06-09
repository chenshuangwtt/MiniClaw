"""Parser for structured agent output.

Takes a raw JSON string from the LLM and returns a validated
``ToolCall`` or ``FinalAnswer`` instance.

Supports multiple input formats:
    - Pure JSON: ``{"type": "final_answer", ...}``
    - Markdown code fence: `` ```json\n{...}\n``` ``
    - Text + JSON: ``Sure! {"type": "final_answer", ...}``
    - Native OpenAI tool_calls (via ``parse_native``)
"""

from __future__ import annotations

import json
import re
from typing import Any

from pydantic import ValidationError

from miniclaw.agent.state import AgentOutput, FinalAnswer, ToolCall


class ParseError(Exception):
    """Raised when the LLM output cannot be parsed or validated.

    Attributes:
        raw: The original string that failed to parse.
        detail: Human-readable explanation of what went wrong.
    """

    def __init__(self, raw: str, detail: str) -> None:
        self.raw = raw
        self.detail = detail
        super().__init__(f"Failed to parse agent output: {detail}")


# Regex for markdown code fences containing JSON
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


class OutputParser:
    """Parses raw LLM JSON into validated ``AgentOutput`` models.

    Usage::

        parser = OutputParser()
        output = parser.parse('{"type": "final_answer", "answer": "Hi!"}')
        assert isinstance(output, FinalAnswer)
    """

    VALID_TYPES = {"tool_call", "final_answer"}

    def parse(self, raw: str) -> AgentOutput:
        """Parse a JSON string into a ``ToolCall`` or ``FinalAnswer``.

        Attempts multiple extraction strategies:
            1. Direct JSON decode.
            2. Extract from markdown code fence.
            3. Find first ``{...}`` in surrounding text.

        Args:
            raw: Raw string from the LLM.

        Returns:
            A ``ToolCall`` or ``FinalAnswer`` instance.

        Raises:
            ParseError: If all extraction strategies fail.
        """
        if not raw or not raw.strip():
            raise ParseError(raw, "Empty response from LLM.")

        stripped = raw.strip()

        # Strategy 1: Direct JSON decode (gives specific errors)
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            # Try fallback strategies before giving up
            data = self._try_extract_code_fence(stripped)
            if data is not None:
                return self._validate(data, stripped)
            data = self._try_extract_first_json(stripped)
            if data is not None:
                return self._validate(data, stripped)
            raise ParseError(stripped, f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ParseError(stripped, f"Expected a JSON object, got {type(data).__name__}.")

        return self._validate(data, stripped)

    def parse_native(
        self,
        content: str | None,
        tool_calls: list[dict[str, Any]] | None = None,
    ) -> AgentOutput:
        """Parse native OpenAI-style response with ``tool_calls`` field.

        Used when the LLM returns structured tool calls natively
        (OpenAI function calling) rather than JSON-in-text.

        Args:
            content: The message content (may be None if only tool_calls).
            tool_calls: List of tool call dicts with ``id``, ``function.name``,
                ``function.arguments``.

        Returns:
            A ``ToolCall`` or ``FinalAnswer``.
        """
        if tool_calls:
            tc = tool_calls[0]
            func = tc.get("function", {})
            name = func.get("name", "unknown")
            raw_args = func.get("arguments", "{}")
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
            except json.JSONDecodeError:
                args = {"_raw": raw_args}
            return ToolCall(tool_name=name, arguments=args, thought=content or "")

        # No tool calls — treat as final answer
        return FinalAnswer(answer=content or "", thought="")

    # ------------------------------------------------------------------
    # Internal extraction strategies
    # ------------------------------------------------------------------

    def _try_parse_json(self, text: str) -> dict[str, Any] | None:
        """Try to parse text as direct JSON."""
        try:
            data = json.loads(text)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _try_extract_code_fence(self, text: str) -> dict[str, Any] | None:
        """Extract JSON from a markdown code fence."""
        match = _CODE_FENCE_RE.search(text)
        if not match:
            return None
        try:
            data = json.loads(match.group(1).strip())
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None

    def _try_extract_first_json(self, text: str) -> dict[str, Any] | None:
        """Find and extract the first {...} JSON object using brace counting."""
        start = text.find("{")
        if start == -1:
            return None
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        data = json.loads(text[start : i + 1])
                        return data if isinstance(data, dict) else None
                    except json.JSONDecodeError:
                        return None
        return None

    def _validate(self, data: dict[str, Any], raw: str) -> AgentOutput:
        """Validate parsed JSON dict against Pydantic models."""
        output_type = data.get("type")
        if output_type is None:
            raise ParseError(raw, 'Missing required field "type".')
        if output_type not in self.VALID_TYPES:
            raise ParseError(
                raw,
                f'Unknown type "{output_type}". Expected one of: {", ".join(sorted(self.VALID_TYPES))}.',
            )
        try:
            if output_type == "tool_call":
                return ToolCall.model_validate(data)
            else:
                return FinalAnswer.model_validate(data)
        except ValidationError as exc:
            raise ParseError(raw, f"Validation error: {exc}") from exc
