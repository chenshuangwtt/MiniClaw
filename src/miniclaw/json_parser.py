"""Robust JSON / tool-call extraction from LLM free-form text."""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from typing import Any

from miniclaw.llm.base import ToolCall


@dataclass
class ParseResult:
    """Result of parsing an LLM response."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0


# Patterns for extracting JSON blocks from markdown
_CODE_FENCE_RE = re.compile(
    r"```(?:json)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE
)
_TOOL_CALL_MARKER = '"tool_call"'


def parse_llm_response(text: str) -> ParseResult:
    """Extract structured data from an LLM's raw response text.

    Handles three common patterns:

    1. **Pure JSON** — the entire response is a JSON object with
       ``tool_call`` or ``tool_calls`` keys.
    2. **Markdown code fences** — JSON inside `` ```json ... ``` ``.
    3. **Embedded tool calls** — a JSON object with ``tool_call`` key
       found anywhere in free-form text.

    If nothing structured is found, returns the original text as-is.

    Args:
        text: Raw string output from the LLM.

    Returns:
        ParseResult with ``text`` and/or ``tool_calls`` populated.
    """
    if not text or not text.strip():
        return ParseResult(text="")

    stripped = text.strip()

    # --- 1. Try parsing the whole response as JSON ---
    result = _try_parse_whole_json(stripped)
    if result is not None:
        return result

    # --- 2. Try extracting from code fences ---
    result = _try_parse_code_fences(stripped)
    if result is not None:
        return result

    # --- 3. Try finding embedded tool_call JSON objects ---
    result = _try_parse_embedded_tool_calls(stripped)
    if result is not None:
        return result

    # --- 4. Nothing structured found; return as plain text ---
    return ParseResult(text=stripped)


# ------------------------------------------------------------------
# Internal strategies
# ------------------------------------------------------------------

def _try_parse_whole_json(text: str) -> ParseResult | None:
    """Attempt to parse *text* as a complete JSON object."""
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    return _extract_from_dict(obj, full_text=text)


def _try_parse_code_fences(text: str) -> ParseResult | None:
    """Look for ```json ... ``` blocks and parse the first one."""
    match = _CODE_FENCE_RE.search(text)
    if not match:
        return None

    json_str = match.group(1).strip()
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    if not isinstance(obj, dict):
        return None

    # Preserve any text outside the code fence
    leading = text[: match.start()].strip()
    trailing = text[match.end() :].strip()
    surrounding = f"{leading}\n{trailing}".strip()

    result = _extract_from_dict(obj)
    if surrounding and not result.text:
        result.text = surrounding
    return result


def _try_parse_embedded_tool_calls(text: str) -> ParseResult | None:
    """Find ``{"tool_call": ...}`` objects embedded in free text.

    Uses brace counting to handle nested JSON objects.
    """
    # Find the marker and walk backward to the opening '{'
    marker_pos = text.find(_TOOL_CALL_MARKER)
    if marker_pos == -1:
        return None

    # Walk backward to find the '{' that starts this JSON object
    start = text.rfind("{", 0, marker_pos)
    if start == -1:
        return None

    # Walk forward with brace counting to find the matching '}'
    depth = 0
    end = -1
    for i in range(start, len(text)):
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        return None

    json_str = text[start : end + 1]
    try:
        obj = json.loads(json_str)
    except json.JSONDecodeError:
        return None

    surrounding = (text[:start] + text[end + 1 :]).strip()

    result = _extract_from_dict(obj)
    if surrounding and not result.text:
        result.text = surrounding
    return result


def _extract_from_dict(
    obj: dict[str, Any], full_text: str = ""
) -> ParseResult:
    """Pull tool_calls out of a parsed JSON dict."""
    tool_calls: list[ToolCall] = []

    # Single tool_call
    if "tool_call" in obj and isinstance(obj["tool_call"], dict):
        tc = obj["tool_call"]
        tool_calls.append(_make_tool_call(tc))

    # Multiple tool_calls
    if "tool_calls" in obj and isinstance(obj["tool_calls"], list):
        for tc in obj["tool_calls"]:
            if isinstance(tc, dict):
                tool_calls.append(_make_tool_call(tc))

    text = obj.get("content", "") or obj.get("text", "") or ""

    return ParseResult(text=text or full_text, tool_calls=tool_calls)


def _make_tool_call(tc: dict[str, Any]) -> ToolCall:
    """Construct a ToolCall from a dict, filling in defaults."""
    return ToolCall(
        id=tc.get("id", f"call_{uuid.uuid4().hex[:8]}"),
        name=tc.get("name", "unknown"),
        arguments=tc.get("arguments", {}),
    )
