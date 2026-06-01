"""Tests for context.py."""

import pytest
from miniclaw.context import ContextManager


class TestContextManager:
    def _make_messages(self, n: int) -> list[dict]:
        msgs = [{"role": "system", "content": "You are helpful."}]
        for i in range(n):
            msgs.append({"role": "user", "content": f"Question {i}"})
            msgs.append({"role": "assistant", "content": f"Answer {i}"})
        return msgs

    def test_no_trimming_when_within_budget(self):
        ctx = ContextManager(max_tokens=10000)
        msgs = self._make_messages(5)
        result = ctx.trim(msgs)
        assert len(result) == len(msgs)

    def test_trims_old_messages_first(self):
        # Very small budget forces trimming
        ctx = ContextManager(max_tokens=20, reserve_tokens=0, pinned_turns=1)
        msgs = self._make_messages(10)
        result = ctx.trim(msgs)
        # Should keep system + last turn
        assert result[0]["role"] == "system"
        assert len(result) < len(msgs)
        # Last message should be preserved
        assert result[-1]["content"] == "Answer 9"

    def test_system_message_always_preserved(self):
        ctx = ContextManager(max_tokens=30, reserve_tokens=0, pinned_turns=1)
        msgs = self._make_messages(20)
        result = ctx.trim(msgs)
        assert result[0]["role"] == "system"
        assert result[0]["content"] == "You are helpful."

    def test_pinned_turns_preserved(self):
        ctx = ContextManager(max_tokens=100, reserve_tokens=0, pinned_turns=2)
        msgs = self._make_messages(10)
        result = ctx.trim(msgs)
        # Last 4 messages (2 turns) should be in result
        assert result[-1]["content"] == "Answer 9"
        assert result[-2]["role"] == "user"
        assert result[-3]["content"] == "Answer 8"

    def test_returns_copy_not_mutate(self):
        ctx = ContextManager(max_tokens=10000)
        msgs = self._make_messages(3)
        result = ctx.trim(msgs)
        result.append({"role": "user", "content": "extra"})
        assert len(msgs) == 7  # unchanged

    def test_custom_count_fn(self):
        """Always report 999 tokens to force trimming."""
        ctx = ContextManager(
            max_tokens=100,
            reserve_tokens=0,
            pinned_turns=1,
            count_fn=lambda msgs: 999,
        )
        msgs = self._make_messages(5)
        result = ctx.trim(msgs)
        assert len(result) < len(msgs)
