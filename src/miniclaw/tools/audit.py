"""In-memory audit log for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any


@dataclass(frozen=True)
class AuditEvent:
    """One tool execution audit record."""

    tool_name: str
    arguments: dict[str, Any]
    allowed: bool
    success: bool
    error: str | None = None
    output_preview: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class AuditLogger:
    """Collects tool execution audit events in memory."""

    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def log(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        allowed: bool,
        success: bool,
        error: str | None = None,
        output: Any = None,
    ) -> None:
        """Append one audit event."""
        preview = "" if output is None else str(output)[:200]
        self._events.append(
            AuditEvent(
                tool_name=tool_name,
                arguments=dict(arguments),
                allowed=allowed,
                success=success,
                error=error,
                output_preview=preview,
            )
        )

    def events(self) -> list[AuditEvent]:
        """Return a copy of collected events."""
        return list(self._events)

    def clear(self) -> None:
        """Clear all audit events."""
        self._events.clear()
