"""Parser for structured agent output.

Takes a raw JSON string from the LLM and returns a validated
``ToolCall`` or ``FinalAnswer`` instance.
"""

from __future__ import annotations

import json
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

        Steps:
            1. JSON-decode the string.
            2. Check that ``type`` is present and valid.
            3. Validate against the appropriate Pydantic model.

        Args:
            raw: Raw JSON string from the LLM.

        Returns:
            A ``ToolCall`` or ``FinalAnswer`` instance.

        Raises:
            ParseError: If JSON is invalid, ``type`` is missing/unknown,
                        or Pydantic validation fails.
        """
        if not raw or not raw.strip():
            raise ParseError(raw, "Empty response from LLM.")

        # Step 1: JSON decode
        try:
            data: dict[str, Any] = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ParseError(raw, f"Invalid JSON: {exc}") from exc

        if not isinstance(data, dict):
            raise ParseError(raw, f"Expected a JSON object, got {type(data).__name__}.")

        # Step 2: Check "type" field
        output_type = data.get("type")
        if output_type is None:
            raise ParseError(raw, 'Missing required field "type".')
        if output_type not in self.VALID_TYPES:
            raise ParseError(
                raw,
                f'Unknown type "{output_type}". Expected one of: {", ".join(sorted(self.VALID_TYPES))}.',
            )

        # Step 3: Validate with Pydantic
        try:
            if output_type == "tool_call":
                return ToolCall.model_validate(data)
            else:
                return FinalAnswer.model_validate(data)
        except ValidationError as exc:
            raise ParseError(raw, f"Validation error: {exc}") from exc
