"""ContextManager — message store with token budgeting and auto-compression.

Responsibilities:
    1. Store conversation messages with typed roles.
    2. Estimate token usage (char count / 4).
    3. Compress old messages into a summary when over budget.
    4. Build the final message list for the LLM.

Design note:
    The ``summarizer`` parameter defaults to a rule-based extractor but can
    be swapped for an LLM-based summarizer without changing any other code.
"""

from __future__ import annotations

from typing import Any, Callable, Literal

# Valid message roles
MessageRole = Literal["system", "user", "assistant", "tool", "summary"]

# Default token budget
DEFAULT_MAX_CONTEXT_TOKENS = 8000
DEFAULT_RECENT_KEEP = 6


class ContextManager:
    """Manages conversation context with automatic compression.

    Args:
        max_context_tokens: Token budget for the entire context.
        recent_keep: Number of recent messages to always keep (never compress).
        summarizer: ``(messages) -> str`` function that produces a summary.
            Defaults to a rule-based extractive summarizer.

    Usage::

        ctx = ContextManager(max_context_tokens=4000)
        ctx.add_message("system", "You are a helpful assistant.")
        ctx.add_message("user", "What's the weather?")
        ctx.add_message("assistant", '{"type": "tool_call", ...}')
        ctx.add_observation("get_weather", {"city": "Beijing"}, "Sunny 25°C")

        if ctx.should_compress():
            ctx.compress()

        messages = ctx.build_messages(tools_prompt="...", current_task="...")
    """

    def __init__(
        self,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        recent_keep: int = DEFAULT_RECENT_KEEP,
        summarizer: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        self.max_context_tokens = max_context_tokens
        self.recent_keep = recent_keep
        self._messages: list[dict[str, Any]] = []
        self._summarizer = summarizer or _rule_based_summarizer

    # ------------------------------------------------------------------
    # Message management
    # ------------------------------------------------------------------

    def add_message(self, role: MessageRole, content: str, **extra: Any) -> None:
        """Append a message to the context.

        Args:
            role: One of ``system``, ``user``, ``assistant``, ``tool``, ``summary``.
            content: The message text.
            **extra: Additional fields (e.g., ``tool_name``, ``tool_call_id``).
        """
        msg: dict[str, Any] = {"role": role, "content": content}
        msg.update(extra)
        self._messages.append(msg)

    def add_observation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        observation: Any,
    ) -> None:
        """Add a tool call result as a structured assistant + tool message pair.

        This inserts two messages:
        1. An ``assistant`` message describing the tool call.
        2. A ``tool`` message with the observation result.
        """
        self.add_message(
            "assistant",
            f"Called {tool_name}({arguments})",
            tool_name=tool_name,
            arguments=arguments,
        )
        self.add_message("tool", str(observation), tool_name=tool_name)

    def get_messages(self) -> list[dict[str, Any]]:
        """Return a shallow copy of all stored messages."""
        return list(self._messages)

    def clear(self) -> None:
        """Remove all messages."""
        self._messages.clear()

    @property
    def message_count(self) -> int:
        """Number of stored messages."""
        return len(self._messages)

    # ------------------------------------------------------------------
    # Token estimation
    # ------------------------------------------------------------------

    def estimate_tokens(self, messages: list[dict[str, Any]] | None = None) -> int:
        """Estimate token count using ``len(content) / 4``.

        Args:
            messages: Messages to estimate.  Defaults to stored messages.
        """
        target = messages if messages is not None else self._messages
        total = 0
        for msg in target:
            content = msg.get("content") or ""
            total += len(str(content))
            # Account for structured fields
            for key in ("tool_calls", "arguments"):
                if key in msg:
                    total += len(str(msg[key]))
        return total // 4

    # ------------------------------------------------------------------
    # Compression
    # ------------------------------------------------------------------

    def should_compress(self) -> bool:
        """True if current token usage exceeds the budget."""
        return self.estimate_tokens() > self.max_context_tokens

    def compress(self) -> bool:
        """Compress old messages into a summary to fit the budget.

        Strategy:
            1. Separate ``system`` messages (always kept).
            2. Keep the last ``recent_keep`` messages untouched.
            3. Summarize everything in between.
            4. Replace the middle section with a single ``summary`` message.

        Returns:
            True if compression was performed, False if already within budget.
        """
        if not self.should_compress():
            return False

        # Separate system messages from the front
        system_msgs: list[dict[str, Any]] = []
        rest = list(self._messages)
        while rest and rest[0].get("role") == "system":
            system_msgs.append(rest.pop(0))

        # If rest is small enough, nothing to compress
        if len(rest) <= self.recent_keep:
            # Already compact — just keep as-is
            self._messages = system_msgs + rest
            return False

        # Split into old (compressible) and recent (pinned)
        old = rest[: -self.recent_keep]
        recent = rest[-self.recent_keep :]

        # Generate summary
        summary_text = self._summarizer(old)

        # Build compressed message list
        compressed: list[dict[str, Any]] = list(system_msgs)
        if summary_text:
            compressed.append({"role": "summary", "content": summary_text})
        compressed.extend(recent)

        self._messages = compressed
        return True

    # ------------------------------------------------------------------
    # Build LLM input
    # ------------------------------------------------------------------

    def build_messages(
        self,
        system_prompt: str = "",
        tools_prompt: str = "",
        long_term_memory: str = "",
        current_task: str = "",
    ) -> list[dict[str, Any]]:
        """Assemble the final message list for the LLM.

        Order:
            1. System prompt
            2. Tools prompt
            3. Long-term memory (if any)
            4. Stored messages (including any summary)
            5. Current task (as the latest user message)

        Args:
            system_prompt: The system-level instructions.
            tools_prompt: Formatted tool descriptions.
            long_term_memory: Persistent memory context.
            current_task: The user's current task/question.

        Returns:
            A list of message dicts ready for ``llm.chat()``.
        """
        result: list[dict[str, Any]] = []

        # 1. System prompt
        if system_prompt:
            result.append({"role": "system", "content": system_prompt})

        # 2. Tools prompt
        if tools_prompt:
            result.append({"role": "system", "content": tools_prompt})

        # 3. Long-term memory
        if long_term_memory:
            result.append({"role": "system", "content": f"[Memory]\n{long_term_memory}"})

        # 4. Stored messages
        result.extend(self._messages)

        # 5. Current task
        if current_task:
            result.append({"role": "user", "content": current_task})

        return result

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._messages)

    def __repr__(self) -> str:
        tokens = self.estimate_tokens()
        return (
            f"<ContextManager messages={len(self._messages)} "
            f"tokens≈{tokens} max={self.max_context_tokens}>"
        )


# ------------------------------------------------------------------
# Default summarizer (rule-based, no LLM call)
# ------------------------------------------------------------------


def _rule_based_summarizer(messages: list[dict[str, Any]]) -> str:
    """Extract key information from messages without calling an LLM.

    Produces a compact summary by:
        - Counting messages by role.
        - Extracting tool names used.
        - Including the last user message and last assistant message.
    """
    if not messages:
        return ""

    role_counts: dict[str, int] = {}
    tool_names: list[str] = []
    last_user = ""
    last_assistant = ""

    for msg in messages:
        role = msg.get("role", "unknown")
        role_counts[role] = role_counts.get(role, 0) + 1

        if role == "assistant":
            last_assistant = str(msg.get("content", ""))[:200]
        elif role == "user":
            last_user = str(msg.get("content", ""))[:200]
        elif role == "tool":
            name = msg.get("tool_name", "unknown")
            if name not in tool_names:
                tool_names.append(name)

    parts = [f"[Compressed {len(messages)} messages]"]
    parts.append(f"Roles: {role_counts}")

    if tool_names:
        parts.append(f"Tools used: {', '.join(tool_names)}")

    if last_user:
        parts.append(f"Last user: {last_user}")
    if last_assistant:
        parts.append(f"Last assistant: {last_assistant}")

    return "\n".join(parts)
