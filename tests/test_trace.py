"""Tests for trace.py."""

import json
import pytest
from pathlib import Path
from miniclaw.trace import TraceLogger


class TestTraceLogger:
    def test_log_event(self):
        t = TraceLogger(console=False)
        t.log("test_event", {"key": "value"})
        events = t.get_events()
        assert len(events) == 1
        assert events[0]["type"] == "test_event"
        assert events[0]["data"]["key"] == "value"
        assert "ts" in events[0]
        assert "run_id" in events[0]

    def test_multiple_events(self):
        t = TraceLogger(console=False)
        t.log("a")
        t.log("b")
        t.log("c")
        assert len(t.get_events()) == 3

    def test_span_logs_start_and_end(self):
        t = TraceLogger(console=False)
        with t.span("llm_call", {"model": "gpt-4"}) as ctx:
            ctx["result"] = "ok"
        events = t.get_events()
        types = [e["type"] for e in events]
        assert "llm_call:start" in types
        assert "llm_call:end" in types
        end_event = [e for e in events if e["type"] == "llm_call:end"][0]
        assert "elapsed_ms" in end_event["data"]
        assert end_event["data"]["result"] == "ok"

    def test_span_captures_exception(self):
        t = TraceLogger(console=False)
        with pytest.raises(ValueError):
            with t.span("boom") as ctx:
                raise ValueError("bad")
        events = t.get_events()
        end_event = [e for e in events if e["type"] == "boom:end"][0]
        assert "bad" in end_event["data"]["error"]

    def test_flush_to_file(self, tmp_path: Path):
        f = tmp_path / "trace.jsonl"
        t = TraceLogger(file_path=f, console=False)
        t.log("event1", {"x": 1})
        t.log("event2", {"x": 2})
        t.flush()

        lines = f.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        obj = json.loads(lines[0])
        assert obj["type"] == "event1"

    def test_get_events_returns_copy(self):
        t = TraceLogger(console=False)
        t.log("a")
        events = t.get_events()
        events.append({"fake": True})
        assert len(t.get_events()) == 1
