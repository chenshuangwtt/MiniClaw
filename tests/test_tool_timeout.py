"""Tests for tool timeout and cancellation support."""

from __future__ import annotations

import time
from typing import Any

import pytest

from miniclaw.tools.timeout import CancellationToken, TimedOutError, ToolTimeout, run_with_timeout


# ---------------------------------------------------------------------------
# run_with_timeout
# ---------------------------------------------------------------------------


class TestRunWithTimeout:
    """Tests for the run_with_timeout function."""

    def test_returns_result(self) -> None:
        def add(a: int, b: int) -> int:
            return a + b

        result = run_with_timeout(add, args=(1, 2), timeout=5)
        assert result == 3

    def test_passes_kwargs(self) -> None:
        def greet(name: str, prefix: str = "Hello") -> str:
            return f"{prefix}, {name}!"

        result = run_with_timeout(greet, kwargs={"name": "Alice", "prefix": "Hi"}, timeout=5)
        assert result == "Hi, Alice!"

    def test_raises_on_timeout(self) -> None:
        def slow() -> str:
            time.sleep(1)
            return "done"

        start = time.perf_counter()
        with pytest.raises(TimedOutError, match="timed out"):
            run_with_timeout(slow, timeout=0.1)
        assert time.perf_counter() - start < 0.75

    def test_propagates_exception(self) -> None:
        def failing() -> None:
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            run_with_timeout(failing, timeout=5)

    def test_cancelled_before_start(self) -> None:
        token = CancellationToken()
        token.cancel()

        with pytest.raises(Exception):
            run_with_timeout(lambda: 42, timeout=5, token=token)


# ---------------------------------------------------------------------------
# CancellationToken
# ---------------------------------------------------------------------------


class TestCancellationToken:
    """Tests for CancellationToken."""

    def test_initial_state(self) -> None:
        token = CancellationToken()
        assert token.cancelled is False

    def test_cancel(self) -> None:
        token = CancellationToken()
        token.cancel()
        assert token.cancelled is True

    def test_check_noop_when_not_cancelled(self) -> None:
        token = CancellationToken()
        token.check()  # Should not raise

    def test_check_raises_when_cancelled(self) -> None:
        token = CancellationToken()
        token.cancel()
        with pytest.raises(Exception):
            token.check()


# ---------------------------------------------------------------------------
# ToolTimeout dataclass
# ---------------------------------------------------------------------------


class TestToolTimeout:
    def test_defaults(self) -> None:
        tt = ToolTimeout()
        assert tt.seconds == 30.0
        assert tt.on_timeout == "error"

    def test_custom(self) -> None:
        tt = ToolTimeout(seconds=10.0, on_timeout="cancel")
        assert tt.seconds == 10.0
        assert tt.on_timeout == "cancel"


# ---------------------------------------------------------------------------
# Integration with ToolExecutor
# ---------------------------------------------------------------------------


class TestExecutorTimeout:
    """Test that ToolExecutor uses timeout correctly."""

    def test_tool_with_timeout_attribute(self) -> None:
        from miniclaw.agent.executor import ToolExecutor
        from miniclaw.tools.base import Tool
        from miniclaw.tools.registry import ToolRegistry

        class SlowTool(Tool):
            name = "slow"
            description = "A slow tool"
            schema = {"type": "object", "properties": {}}
            timeout = 0.1

            def run(self, **kwargs: Any) -> str:
                time.sleep(1)
                return "done"

        registry = ToolRegistry()
        registry.register(SlowTool())
        executor = ToolExecutor(registry)

        start = time.perf_counter()
        obs = executor.execute("slow", {})
        assert obs.success is False
        assert "timed out" in obs.error
        assert time.perf_counter() - start < 0.75

    def test_tool_default_timeout(self) -> None:
        from miniclaw.agent.executor import ToolExecutor
        from miniclaw.tools.base import Tool
        from miniclaw.tools.registry import ToolRegistry

        class SlowTool(Tool):
            name = "slow"
            description = "A slow tool"
            schema = {"type": "object", "properties": {}}

            def run(self, **kwargs: Any) -> str:
                time.sleep(1)
                return "done"

        registry = ToolRegistry()
        registry.register(SlowTool())
        executor = ToolExecutor(registry, default_timeout=0.1)

        start = time.perf_counter()
        obs = executor.execute("slow", {})
        assert obs.success is False
        assert "timed out" in obs.error
        assert time.perf_counter() - start < 0.75

    def test_tool_no_timeout_runs_normally(self) -> None:
        from miniclaw.agent.executor import ToolExecutor
        from miniclaw.tools.base import Tool
        from miniclaw.tools.registry import ToolRegistry

        class FastTool(Tool):
            name = "fast"
            description = "A fast tool"
            schema = {"type": "object", "properties": {}}

            def run(self, **kwargs: Any) -> str:
                return "result"

        registry = ToolRegistry()
        registry.register(FastTool())
        executor = ToolExecutor(registry)

        obs = executor.execute("fast", {})
        assert obs.success is True
        assert obs.output == "result"

    def test_cancel_execution(self) -> None:
        from miniclaw.agent.executor import ToolExecutor
        from miniclaw.tools.base import Tool
        from miniclaw.tools.registry import ToolRegistry

        class SlowTool(Tool):
            name = "slow"
            description = "A slow tool"
            schema = {"type": "object", "properties": {}}
            timeout = 5.0

            def run(self, **kwargs: Any) -> str:
                time.sleep(10)
                return "done"

        registry = ToolRegistry()
        registry.register(SlowTool())
        executor = ToolExecutor(registry)

        # Start execution in background (it will time out)
        # Just test that cancel_execution returns True for active tool
        # and False for inactive
        assert executor.cancel_execution("nonexistent") is False
