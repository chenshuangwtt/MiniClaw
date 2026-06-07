"""Context window manager — keep messages within the model's token budget."""

from __future__ import annotations

from typing import Any, Callable


class ContextManager:
    """Manages the messages list to stay within a token budget.

    Strategies (applied in order when over budget):
    1. Remove oldest non-system, non-pinned messages.
    2. If a summarizer is provided, replace removed messages with a summary.

    Attributes:
        max_tokens: Token budget for the messages list.
        reserve_tokens: Tokens reserved for the LLM's response.
        pinned_turns: Number of recent user/assistant turns that must not be cut.
    """

    def __init__(
        self,
        max_tokens: int = 8000,
        reserve_tokens: int = 1024,
        pinned_turns: int = 2,
        count_fn: Callable[[list[dict[str, Any]]], int] | None = None,
        summarizer: Callable[[list[dict[str, Any]]], str] | None = None,
    ) -> None:
        self.max_tokens = max_tokens
        self.reserve_tokens = reserve_tokens
        self.pinned_turns = pinned_turns
        self._count_fn = count_fn or _default_count
        self._summarizer = summarizer

    def trim(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Return a trimmed copy of *messages* that fits within the budget.

        The system message (first message, if role == "system") and the
        last ``pinned_turns`` user/assistant turn-pairs are always kept.
        """
        available = self.max_tokens - self.reserve_tokens
        if self._count_fn(messages) <= available:
            return list(messages)

        # Separate system message
        system_msgs: list[dict[str, Any]] = []
        rest = list(messages)
        if rest and rest[0].get("role") == "system":
            system_msgs = [rest.pop(0)]

        # Identify pinned tail (last N turn-pairs)
        pinned_count = self._pinned_message_count(rest)
        pinned = rest[-pinned_count:] if pinned_count else []
        trimmable = rest[:-pinned_count] if pinned_count else rest

        # Drop trimmable messages from oldest to newest until we fit
        for drop_count in range(len(trimmable) + 1):
            kept_trimmable = trimmable[drop_count:]
            candidate = system_msgs + kept_trimmable + pinned
            if self._count_fn(candidate) <= available:
                break
        else:
            # Even with all trimmable dropped, still over budget
            kept_trimmable = []

        dropped = trimmable[: len(trimmable) - len(kept_trimmable)]

        # Optionally summarize dropped messages
        if dropped and self._summarizer:
            summary = self._summarizer(dropped)
            kept_trimmable = [
                {"role": "system", "content": f"[Summary of earlier context]: {summary}"}
            ] + kept_trimmable

        return system_msgs + kept_trimmable + pinned

    def _pinned_message_count(self, messages: list[dict[str, Any]]) -> int:
        """Count messages in the last N turn-pairs from the tail."""
        count = 0
        turns_seen = 0
        for msg in reversed(messages):
            count += 1
            if msg.get("role") == "user":
                turns_seen += 1
                if turns_seen >= self.pinned_turns:
                    break
        return count


def _default_count(messages: list[dict[str, Any]]) -> int:
    """Rough token estimate: ~4 chars per token."""
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        total += len(str(content)) // 4
        if "tool_calls" in msg:
            total += len(str(msg["tool_calls"])) // 4
    return total
