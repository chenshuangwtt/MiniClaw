"""Minimal secret redaction for persistent runtime diagnostics.

This is deliberately conservative: it removes common credential values from
traces, audit records, and checkpoints without claiming to be a DLP system.
"""

from __future__ import annotations

import json
import re
from typing import Any


_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|token|password|secret)", re.I)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:api[_-]?key|token|password|secret)\s*[=:]\s*[^\s,;]+)"
)
_SENSITIVE_ASSIGNMENT = re.compile(
    r"(?i)(?P<prefix>['\"]?(?:api[_-]?key|authorization|token|password|secret)['\"]?\s*[:=]\s*)"
    r"(?P<quote>['\"]?)(?P<value>[^\s,;\}\]\'\"]+)(?P=quote)"
)
_REDACTED = "[REDACTED]"


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted copy suitable for persistence or display.

    JSON-encoded strings are decoded and redacted structurally before being
    serialized again. The caller's runtime object is never mutated.
    """
    if isinstance(value, str):
        parsed = _parse_json_container(value)
        if parsed is not None:
            if contains_sensitive_data(parsed):
                return json.dumps(redact_secrets(parsed), ensure_ascii=False, separators=(",", ":"))
            return value
        return _redact_text(value)
    if isinstance(value, dict):
        return {
            key: _REDACTED if _SENSITIVE_KEY.search(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(item) for item in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(item) for item in value)
    return value


def contains_sensitive_data(value: Any) -> bool:
    """Return whether *value* contains a sensitive field name or credential text."""
    if isinstance(value, str):
        parsed = _parse_json_container(value)
        return (
            contains_sensitive_data(parsed)
            if parsed is not None
            else bool(_SENSITIVE_VALUE.search(value) or _SENSITIVE_ASSIGNMENT.search(value))
        )
    if isinstance(value, dict):
        return any(
            _SENSITIVE_KEY.search(str(key)) or contains_sensitive_data(item)
            for key, item in value.items()
        )
    if isinstance(value, (list, tuple)):
        return any(contains_sensitive_data(item) for item in value)
    return False


def _parse_json_container(value: str) -> dict[str, Any] | list[Any] | None:
    """Decode only complete JSON objects/arrays, leaving ordinary text intact."""
    stripped = value.strip()
    if not stripped.startswith(("{", "[")):
        return None
    try:
        parsed = json.loads(stripped)
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, (dict, list)) else None


def _redact_text(value: str) -> str:
    """Redact credential values while preserving surrounding diagnostic text."""
    value = _SENSITIVE_VALUE.sub(_REDACTED, value)

    def replace_assignment(match: re.Match[str]) -> str:
        quote = match.group("quote")
        return f"{match.group('prefix')}{quote}{_REDACTED}{quote}"

    return _SENSITIVE_ASSIGNMENT.sub(replace_assignment, value)
