"""Tests for TraceLogger HTML report generation."""

from __future__ import annotations

import tempfile
from pathlib import Path


from miniclaw.agent.trace import TraceLogger, _action_badge, _escape_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestEscapeHtml:
    def test_basic(self) -> None:
        assert _escape_html("<script>") == "&lt;script&gt;"

    def test_ampersand(self) -> None:
        assert _escape_html("a & b") == "a &amp; b"

    def test_quotes(self) -> None:
        assert _escape_html('"hello"') == "&quot;hello&quot;"


class TestActionBadge:
    def test_tool_call(self) -> None:
        badge = _action_badge("tool_call")
        assert "tool_call" in badge
        assert "badge-tool" in badge

    def test_final_answer(self) -> None:
        badge = _action_badge("final_answer")
        assert "final_answer" in badge
        assert "badge-final" in badge

    def test_unknown(self) -> None:
        badge = _action_badge("custom_action")
        assert "custom_action" in badge


# ---------------------------------------------------------------------------
# TraceLogger.to_html
# ---------------------------------------------------------------------------


class TestTraceToHtml:
    def _make_trace(self) -> TraceLogger:
        """Create a sample TraceLogger with a few steps."""
        trace = TraceLogger()
        trace.log_step(
            step=1,
            model_output='{"type": "tool_call", "tool_name": "read_file"}',
            parsed_action="tool_call",
            tool_name="read_file",
            arguments={"path": "README.md"},
            observation="File contents here...",
        )
        trace.log_step(
            step=2,
            model_output='{"type": "final_answer"}',
            parsed_action="final_answer",
            observation="Here is the analysis.",
        )
        return trace

    def test_html_contains_title(self) -> None:
        trace = self._make_trace()
        html = trace.to_html("My Report")
        assert "My Report" in html

    def test_html_contains_summary(self) -> None:
        trace = self._make_trace()
        html = trace.to_html()
        assert "Steps" in html
        assert "Tool Calls" in html

    def test_html_contains_step_cards(self) -> None:
        trace = self._make_trace()
        html = trace.to_html()
        assert "Step 1" in html
        assert "Step 2" in html
        assert "read_file" in html

    def test_html_contains_mermaid(self) -> None:
        trace = self._make_trace()
        html = trace.to_html()
        assert "flowchart TD" in html

    def test_html_is_valid_structure(self) -> None:
        trace = self._make_trace()
        html = trace.to_html()
        assert html.startswith("<!DOCTYPE html>")
        assert "</html>" in html
        assert "<style>" in html

    def test_html_with_error_step(self) -> None:
        trace = TraceLogger()
        trace.log_step(
            step=1,
            parsed_action="tool_call",
            tool_name="broken_tool",
            arguments={"x": 1},
            error="Tool failed: connection timeout",
        )
        html = trace.to_html()
        assert "connection timeout" in html
        assert "error-block" in html

    def test_html_empty_trace(self) -> None:
        trace = TraceLogger()
        html = trace.to_html("Empty")
        assert "Empty" in html
        assert "0" in html


# ---------------------------------------------------------------------------
# TraceLogger.export_html
# ---------------------------------------------------------------------------


class TestExportHtml:
    def test_export_creates_file(self) -> None:
        trace = TraceLogger()
        trace.log_step(step=1, parsed_action="final_answer", observation="done")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as f:
            path = f.name

        try:
            trace.export_html(path, title="Test Export")
            content = Path(path).read_text(encoding="utf-8")
            assert "<!DOCTYPE html>" in content
            assert "Test Export" in content
        finally:
            Path(path).unlink(missing_ok=True)

    def test_export_creates_parent_dirs(self) -> None:
        trace = TraceLogger()
        trace.log_step(step=1, parsed_action="final_answer", observation="done")

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "sub" / "report.html"
            trace.export_html(str(path))
            assert path.exists()
