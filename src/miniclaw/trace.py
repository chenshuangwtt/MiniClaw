"""Structured trace logger for LLM calls and tool executions."""

from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Generator

logger = logging.getLogger("miniclaw.trace")


class TraceLogger:
    """Records structured events to the console and/or a JSON-lines file.

    Events are simple dicts with at minimum ``type``, ``ts``, and ``data``.

    Usage::

        trace = TraceLogger("run.trace.jsonl")
        trace.log("llm_call", {"messages": msgs, "response": resp})
        trace.log("tool_call", {"name": "get_weather", "args": {...}, "result": ...})
        trace.flush()
    """

    def __init__(
        self,
        file_path: str | Path | None = None,
        console: bool = True,
    ) -> None:
        self._file_path = Path(file_path) if file_path else None
        self._console = console
        self._events: list[dict[str, Any]] = []
        self._run_id = uuid.uuid4().hex[:12]

        if self._file_path:
            self._file_path.parent.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Core logging
    # ------------------------------------------------------------------

    def log(self, event_type: str, data: dict[str, Any] | None = None) -> None:
        """Record a trace event (buffered; call flush() to write to file)."""
        event = {
            "type": event_type,
            "ts": time.time(),
            "run_id": self._run_id,
            "data": data or {},
        }
        self._events.append(event)

        if self._console:
            self._print_event(event)

    @contextmanager
    def span(self, event_type: str, data: dict[str, Any] | None = None) -> Generator[dict[str, Any], None, None]:
        """Context manager that logs ``start`` and ``end`` with elapsed time.

        Usage::

            with trace.span("llm_call", {"model": "gpt-4o"}) as ctx:
                resp = llm.chat(messages)
                ctx["response"] = resp
        """
        ctx = dict(data or {})
        self.log(f"{event_type}:start", ctx)
        t0 = time.monotonic()
        try:
            yield ctx
        except Exception as exc:
            ctx["error"] = str(exc)
            raise
        finally:
            ctx["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)
            self.log(f"{event_type}:end", ctx)

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """Write all buffered events to the file (if configured)."""
        if not self._file_path or not self._events:
            return
        with open(self._file_path, "a", encoding="utf-8") as f:
            for event in self._events:
                f.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")
        self._events.clear()

    def get_events(self) -> list[dict[str, Any]]:
        """Return a copy of all recorded events."""
        return list(self._events)

    def _print_event(self, event: dict[str, Any]) -> None:
        etype = event["type"]
        data = event.get("data", {})
        # Compact one-line summary for the terminal
        summary = _compact_summary(data)
        logger.info("⚡ %s %s", etype, summary)


def _compact_summary(data: dict[str, Any]) -> str:
    """Build a short human-readable summary from event data."""
    parts: list[str] = []
    for key in ("name", "model", "elapsed_ms", "error"):
        if key in data:
            parts.append(f"{key}={data[key]}")
    if not parts:
        # fallback: first 120 chars of JSON
        raw = json.dumps(data, ensure_ascii=False, default=str)
        return raw[:120] + ("…" if len(raw) > 120 else "")
    return " ".join(parts)
