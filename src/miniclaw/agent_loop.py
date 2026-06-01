"""Agent Loop — the core runtime that drives LLM ↔ Tool interactions."""

from __future__ import annotations

import logging
from typing import Any

from miniclaw.context import ContextManager
from miniclaw.json_parser import parse_llm_response
from miniclaw.llm.base import BaseLLM, LLMResponse
from miniclaw.memory import Memory
from miniclaw.recovery import RecoveryManager
from miniclaw.tool_registry import ToolRegistry
from miniclaw.trace import TraceLogger

logger = logging.getLogger(__name__)

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "When you need to call a tool, respond with a JSON object: "
    '{"tool_call": {"name": "<tool_name>", "arguments": {<args>}}}'
)


class Agent:
    """Orchestrates the LLM ↔ Tool conversation loop.

    Flow per turn:
        1. Build messages (system + history + user input).
        2. Trim to fit context window via ContextManager.
        3. Call LLM via RecoveryManager (retry on failure).
        4. Parse response via json_parser.
        5. If tool_calls → execute tools → append results → goto 3.
        6. If text → return to caller.

    Attributes:
        max_turns: Maximum tool-call rounds before forced stop.
    """

    def __init__(
        self,
        llm: BaseLLM,
        tools: ToolRegistry | None = None,
        context: ContextManager | None = None,
        recovery: RecoveryManager | None = None,
        memory: Memory | None = None,
        trace: TraceLogger | None = None,
        system_prompt: str = DEFAULT_SYSTEM_PROMPT,
        max_turns: int = 10,
    ) -> None:
        self.llm = llm
        self.tools = tools or ToolRegistry()
        self.context = context or ContextManager()
        self.recovery = recovery or RecoveryManager()
        self.memory = memory
        self.trace = trace
        self.system_prompt = system_prompt
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, user_input: str) -> str:
        """Run one complete interaction: user input → final answer.

        Returns the assistant's final text response.
        """
        messages = self._build_messages(user_input)

        for turn in range(self.max_turns):
            messages = self.context.trim(messages)

            # --- Call LLM ---
            llm_response = self._call_llm(messages)
            if self.trace:
                self.trace.log("llm_response", {
                    "turn": turn,
                    "content": llm_response.content[:200],
                    "tool_call_count": len(llm_response.tool_calls),
                })

            # --- Check for tool calls ---
            if not llm_response.tool_calls:
                # Also check the text content for embedded tool calls
                parsed = parse_llm_response(llm_response.content)
                if not parsed.has_tool_calls:
                    # Pure text — we're done
                    final = llm_response.content or parsed.text
                    self._save_conversation(user_input, final)
                    return final
                tool_calls = parsed.tool_calls
                # Keep any surrounding text as the assistant's message
                if parsed.text:
                    messages.append({"role": "assistant", "content": parsed.text})
            else:
                tool_calls = llm_response.tool_calls
                if llm_response.content:
                    messages.append({"role": "assistant", "content": llm_response.content})

            # --- Execute tools ---
            for tc in tool_calls:
                if self.trace:
                    self.trace.log("tool_call:start", {"name": tc.name, "args": tc.arguments})
                result = self._execute_tool(tc.name, tc.arguments)
                result_str = str(result)
                if self.trace:
                    self.trace.log("tool_call:end", {"name": tc.name, "result": result_str[:200]})
                # Append tool call + result to messages
                messages.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [{"id": tc.id, "type": "function", "function": {"name": tc.name, "arguments": _safe_json(tc.arguments)}}],
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result_str,
                })

        # Max turns exceeded
        logger.warning("Agent loop exceeded %d turns.", self.max_turns)
        return "[Agent] Maximum tool-call rounds exceeded."

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _build_messages(self, user_input: str) -> list[dict[str, Any]]:
        """Assemble the messages list: system + memory history + new user msg."""
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        if self.memory:
            history = self.memory.get_messages(limit=20)
            for msg in history:
                messages.append({"role": msg["role"], "content": msg["content"]})
        messages.append({"role": "user", "content": user_input})
        return messages

    def _call_llm(self, messages: list[dict[str, Any]]) -> LLMResponse:
        """Call the LLM with retry logic."""
        tools_schema = self.tools.to_openai_tools() if self.tools else None

        def _do_call() -> LLMResponse:
            return self.llm.chat(messages, tools=tools_schema or None)

        return self.recovery.call_with_retry(_do_call)

    def _execute_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Execute a tool, catching and returning errors as strings."""
        try:
            return self.tools.execute(name, arguments)
        except KeyError:
            return f"Error: tool '{name}' is not registered."
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", name)
            return f"Error executing {name}: {exc}"

    def _save_conversation(self, user_input: str, assistant_reply: str) -> None:
        """Persist the turn to memory (if configured)."""
        if self.memory:
            self.memory.append_message("user", user_input)
            self.memory.append_message("assistant", assistant_reply)


def _safe_json(obj: Any) -> str:
    """Serialize to JSON, falling back to str on failure."""
    import json
    try:
        return json.dumps(obj, ensure_ascii=False)
    except Exception:
        return str(obj)
