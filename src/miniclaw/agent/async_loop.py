"""Asynchronous compatibility API for the shared agent engine."""

from __future__ import annotations

from miniclaw.agent.engine import AgentEngine, AgentResult


DEFAULT_MAX_STEPS = 10
DEFAULT_MAX_ERRORS = 3


class AsyncAgentLoop(AgentEngine):
    """Async facade over the same runtime used by :class:`AgentLoop`."""

    async def run(self, user_task: str, user_id: str = "default") -> AgentResult:
        """Execute a task without blocking the event loop."""
        return await self.run_async(user_task, user_id)
