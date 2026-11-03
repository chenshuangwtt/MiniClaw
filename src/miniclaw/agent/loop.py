"""Synchronous compatibility API for the shared agent engine."""

from __future__ import annotations

import asyncio

from miniclaw.agent.engine import AgentEngine, AgentResult


DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_ERRORS = 3


class AgentLoop(AgentEngine):
    """Run :class:`AgentEngine` from ordinary synchronous Python code.

    The execution logic intentionally lives in ``AgentEngine`` so this public
    API and ``AsyncAgentLoop`` cannot drift apart.
    """

    def run(self, user_task: str, user_id: str = "default") -> AgentResult:
        """Execute a task and return its complete result."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.run_async(user_task, user_id))
        raise RuntimeError(
            "AgentLoop.run() cannot be called inside an active event loop; "
            "use AsyncAgentLoop.run() instead."
        )
