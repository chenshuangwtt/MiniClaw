"""Tests for recovery.py."""

import pytest
from miniclaw.recovery import RecoveryManager


class TestRecoveryManager:
    def test_success_on_first_try(self):
        rm = RecoveryManager(max_retries=3, backoff_base=0.01)
        result = rm.call_with_retry(lambda: 42)
        assert result == 42

    def test_retries_on_failure_then_succeeds(self):
        call_count = 0

        def flaky():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("oops")
            return "ok"

        rm = RecoveryManager(max_retries=3, backoff_base=0.01)
        result = rm.call_with_retry(flaky)
        assert result == "ok"
        assert call_count == 3

    def test_raises_after_max_retries(self):
        rm = RecoveryManager(max_retries=2, backoff_base=0.01)
        with pytest.raises(ValueError, match="permanent"):
            rm.call_with_retry(lambda: (_ for _ in ()).throw(ValueError("permanent")))

    def test_passes_args_and_kwargs(self):
        rm = RecoveryManager()

        def add(a, b, c=0):
            return a + b + c

        assert rm.call_with_retry(add, 1, 2, c=10) == 13

    def test_get_repair_messages(self):
        rm = RecoveryManager()
        original = [{"role": "user", "content": "hi"}]
        repair = rm.get_repair_messages(original, "bad output")
        assert len(repair) == 3
        assert repair[1]["role"] == "assistant"
        assert repair[1]["content"] == "bad output"
        assert repair[2]["role"] == "user"
        assert "parsed" in repair[2]["content"].lower()
        # Original not mutated
        assert len(original) == 1
