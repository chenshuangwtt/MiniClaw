"""Minimal secret redaction for persistent runtime diagnostics.

This is deliberately conservative: it removes common credential values from
traces, audit records, and checkpoints without claiming to be a DLP system.
"""

from __future__ import annotations

import re
from typing import Any


_SENSITIVE_KEY = re.compile(r"(?:api[_-]?key|authorization|token|password|secret)", re.I)
_SENSITIVE_VALUE = re.compile(
    r"(?i)(?:sk-[A-Za-z0-9_-]{8,}|bearer\s+[A-Za-z0-9._~+/=-]{8,}|"
    r"(?:api[_-]?key|token|password|secret)\s*[=:]\s*[^\s,;]+)"
)
_REDACTED = "[REDACTED]"


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted copy suitable for local diagnostics."""
    if isinstance(value, str):
        return _SENSITIVE_VALUE.sub(_REDACTED, value)
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
