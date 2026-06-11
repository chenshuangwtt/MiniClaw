"""Structured trace logger for agent loop steps.

Each step in the agent loop produces a ``StepTrace`` that records
everything that happened: the LLM output, parsed action, tool execution,
and any errors.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field


class StepTrace(BaseModel):
    """A single step's trace record.

    Attributes:
        step: Step number (1-indexed).
        timestamp: Unix timestamp of this step.
        model_output: Raw string from the LLM.
        parsed_action: "tool_call" or "final_answer" or "parse_error".
        tool_name: Name of the tool called (if tool_call).
        arguments: Arguments passed to the tool (if tool_call).
        observation: Tool execution result (if tool_call).
        error: Error message, if any.
    """

    step: int
    timestamp: float = Field(default_factory=time.time)
    model_output: str = ""
    parsed_action: str = ""
    tool_name: str | None = None
    arguments: dict[str, Any] | None = None
    observation: Any = None
    error: str | None = None


class TraceLogger:
    """Collects ``StepTrace`` records for an agent run.

    Usage::

        trace = TraceLogger()
        trace.log_step(step=1, model_output='{"type": "tool_call", ...}', ...)
        trace.log_step(step=2, model_output='{"type": "final_answer", ...}', ...)

        # Access traces
        for t in trace.steps:
            print(t.step, t.parsed_action)

        # Export to JSONL
        trace.export_jsonl("run.trace.jsonl")
    """

    def __init__(self) -> None:
        self.steps: list[StepTrace] = []

    def log_step(
        self,
        step: int,
        model_output: str = "",
        parsed_action: str = "",
        tool_name: str | None = None,
        arguments: dict[str, Any] | None = None,
        observation: Any = None,
        error: str | None = None,
    ) -> StepTrace:
        """Record a trace for one agent step.

        Returns:
            The created ``StepTrace``.
        """
        trace = StepTrace(
            step=step,
            model_output=model_output,
            parsed_action=parsed_action,
            tool_name=tool_name,
            arguments=arguments,
            observation=observation,
            error=error,
        )
        self.steps.append(trace)
        return trace

    def export_jsonl(self, path: str | Path) -> None:
        """Write all traces to a JSONL file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            for trace in self.steps:
                f.write(trace.model_dump_json() + "\n")

    def to_dicts(self) -> list[dict[str, Any]]:
        """Return all traces as plain dicts."""
        return [t.model_dump() for t in self.steps]

    def to_html(self, title: str = "Agent Trace") -> str:
        """Generate a self-contained HTML report from the trace steps.

        The report includes:
            - A summary bar (total steps, successes, errors, duration)
            - A Mermaid flowchart diagram
            - A timeline of step cards with expandable details

        Args:
            title: Report title.

        Returns:
            A complete HTML string (no external dependencies except
            mermaid.js from CDN).
        """
        total = len(self.steps)
        errors = sum(1 for s in self.steps if s.error)
        tool_calls = sum(1 for s in self.steps if s.parsed_action == "tool_call")
        final_answers = sum(1 for s in self.steps if s.parsed_action == "final_answer")

        duration = 0.0
        if len(self.steps) >= 2:
            duration = self.steps[-1].timestamp - self.steps[0].timestamp

        mermaid_chart = self.to_mermaid(title)
        escaped_mermaid = mermaid_chart.replace("<", "&lt;").replace(">", "&gt;")

        step_cards = []
        for s in self.steps:
            card = self._render_step_card(s)
            step_cards.append(card)

        html = _HTML_TEMPLATE.format(
            title=_escape_html(title),
            total=total,
            tool_calls=tool_calls,
            final_answers=final_answers,
            errors=errors,
            duration=f"{duration:.1f}",
            mermaid_chart=escaped_mermaid,
            step_cards="\n".join(step_cards),
        )
        return html

    def export_html(self, path: str | Path, title: str = "Agent Trace") -> None:
        """Write the trace report to an HTML file.

        Args:
            path: Output file path.
            title: Report title.
        """
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(self.to_html(title), encoding="utf-8")

    def _render_step_card(self, step: StepTrace) -> str:
        """Render a single step as an HTML card."""
        action_badge = _action_badge(step.parsed_action)
        tool_info = ""
        if step.tool_name:
            tool_info = f'<span class="tool-name">{_escape_html(step.tool_name)}</span>'

        error_block = ""
        if step.error:
            error_block = f'<div class="error-block">❌ {_escape_html(step.error[:200])}</div>'

        observation_block = ""
        if step.observation is not None:
            obs_str = str(step.observation)[:500]
            observation_block = (
                f"<details><summary>Observation</summary>"
                f"<pre>{_escape_html(obs_str)}</pre></details>"
            )

        arguments_block = ""
        if step.arguments:
            import json

            try:
                args_str = json.dumps(step.arguments, indent=2, ensure_ascii=False)[:500]
            except Exception:
                args_str = str(step.arguments)[:500]
            arguments_block = (
                f"<details><summary>Arguments</summary>"
                f"<pre>{_escape_html(args_str)}</pre></details>"
            )

        return f"""
        <div class="step-card">
            <div class="step-header">
                <span class="step-num">Step {step.step}</span>
                {action_badge}
                {tool_info}
            </div>
            {arguments_block}
            {observation_block}
            {error_block}
        </div>"""

    def to_mermaid(self, title: str = "Agent Trace") -> str:
        """Generate a Mermaid flowchart from the trace steps.

        Renders as a top-to-bottom flowchart showing each step's action,
        tool call, observation, and errors.

        Returns:
            A Mermaid diagram string (can be embedded in markdown).
        """
        lines = ["flowchart TD"]
        lines.append(f'    Start["{self._escape(title)}"]')

        prev_node = "Start"
        for t in self.steps:
            node_id = f"S{t.step}"

            if t.parsed_action == "final_answer":
                label = f"✅ Final Answer\\n{self._escape(str(t.observation)[:60])}"
                lines.append(f'    {node_id}["{label}"]')
                lines.append(f"    {prev_node} --> {node_id}")
                lines.append(f"    style {node_id} fill:#90EE90")
                break

            elif t.parsed_action == "tool_call":
                label = f"🔧 {t.tool_name}"
                if t.arguments:
                    args_str = ", ".join(f"{k}={v}" for k, v in list(t.arguments.items())[:2])
                    label += f"\\n({self._escape(args_str)})"
                lines.append(f'    {node_id}["{label}"]')
                lines.append(f"    {prev_node} --> {node_id}")

                # Observation node
                obs_id = f"O{t.step}"
                if t.error:
                    obs_label = f"❌ {self._escape(t.error[:50])}"
                    lines.append(f'    {obs_id}["{obs_label}"]')
                    lines.append(f"    {node_id} --> {obs_id}")
                    lines.append(f"    style {obs_id} fill:#FFB3BA")
                elif t.observation is not None:
                    obs_str = str(t.observation)[:50]
                    obs_label = f"→ {self._escape(obs_str)}"
                    lines.append(f'    {obs_id}["{obs_label}"]')
                    lines.append(f"    {node_id} --> {obs_id}")
                    lines.append(f"    style {obs_id} fill:#B3D9FF")

                prev_node = node_id

            elif t.parsed_action == "parse_error":
                label = f"⚠️ Parse Error\\n{self._escape(t.error or '')[:40]}"
                lines.append(f'    {node_id}["{label}"]')
                lines.append(f"    {prev_node} --> {node_id}")
                lines.append(f"    style {node_id} fill:#FFE0B2")
                prev_node = node_id

            else:
                label = f"Step {t.step}: {t.parsed_action}"
                lines.append(f'    {node_id}["{self._escape(label)}"]')
                lines.append(f"    {prev_node} --> {node_id}")
                prev_node = node_id

        lines.append("    style Start fill:#E8E8E8")
        return "\n".join(lines)

    @staticmethod
    def _escape(text: str) -> str:
        """Escape special characters for Mermaid labels."""
        return text.replace('"', "'").replace("\n", " ").replace("\\", "")

    def __len__(self) -> int:
        return len(self.steps)


# ---------------------------------------------------------------------------
# HTML report helpers
# ---------------------------------------------------------------------------


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    )


def _action_badge(action: str) -> str:
    """Return an HTML badge for the action type."""
    badges = {
        "tool_call": '<span class="badge badge-tool">🔧 tool_call</span>',
        "final_answer": '<span class="badge badge-final">✅ final_answer</span>',
        "parse_error": '<span class="badge badge-error">⚠️ parse_error</span>',
        "llm_error": '<span class="badge badge-error">💥 llm_error</span>',
    }
    return badges.get(action, f'<span class="badge">{_escape_html(action)}</span>')


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
         background: #f5f5f5; color: #333; padding: 20px; }}
  .container {{ max-width: 900px; margin: 0 auto; }}
  h1 {{ font-size: 1.5em; margin-bottom: 16px; }}
  .summary {{ display: flex; gap: 16px; margin-bottom: 24px; flex-wrap: wrap; }}
  .summary-card {{ background: #fff; border-radius: 8px; padding: 16px 20px;
                   box-shadow: 0 1px 3px rgba(0,0,0,0.1); min-width: 120px; text-align: center; }}
  .summary-card .num {{ font-size: 2em; font-weight: 700; }}
  .summary-card .label {{ font-size: 0.85em; color: #666; }}
  .mermaid-box {{ background: #fff; border-radius: 8px; padding: 16px;
                  box-shadow: 0 1px 3px rgba(0,0,0,0.1); margin-bottom: 24px;
                  overflow-x: auto; }}
  .step-card {{ background: #fff; border-radius: 8px; padding: 12px 16px;
                margin-bottom: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.08); }}
  .step-header {{ display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }}
  .step-num {{ font-weight: 700; font-size: 0.9em; color: #555; }}
  .tool-name {{ background: #e8f0fe; padding: 2px 8px; border-radius: 4px;
                font-family: monospace; font-size: 0.85em; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.8em; font-weight: 600; }}
  .badge-tool {{ background: #e3f2fd; color: #1565c0; }}
  .badge-final {{ background: #e8f5e9; color: #2e7d32; }}
  .badge-error {{ background: #fff3e0; color: #e65100; }}
  details {{ margin: 6px 0; }}
  details summary {{ cursor: pointer; font-size: 0.85em; color: #666; }}
  pre {{ background: #f8f8f8; padding: 8px; border-radius: 4px;
         font-size: 0.8em; overflow-x: auto; white-space: pre-wrap;
         word-break: break-word; margin-top: 4px; }}
  .error-block {{ background: #ffebee; color: #c62828; padding: 8px;
                  border-radius: 4px; font-size: 0.85em; margin-top: 6px; }}
</style>
</head>
<body>
<div class="container">
  <h1>🐾 {title}</h1>
  <div class="summary">
    <div class="summary-card"><div class="num">{total}</div><div class="label">Steps</div></div>
    <div class="summary-card"><div class="num">{tool_calls}</div><div class="label">Tool Calls</div></div>
    <div class="summary-card"><div class="num">{final_answers}</div><div class="label">Answers</div></div>
    <div class="summary-card"><div class="num">{errors}</div><div class="label">Errors</div></div>
    <div class="summary-card"><div class="num">{duration}s</div><div class="label">Duration</div></div>
  </div>
  <div class="mermaid-box">
    <pre class="mermaid">{mermaid_chart}</pre>
  </div>
  <h2 style="margin-bottom:12px;">Timeline</h2>
  {step_cards}
</div>
<script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{startOnLoad:true, theme:"default"}});</script>
</body>
</html>
"""
