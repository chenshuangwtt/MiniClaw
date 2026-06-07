"""ToolExecutor — safe dispatch of tool calls with structured observations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class Observation(BaseModel):
    """Structured result returned after executing a tool.

    Attributes:
        tool_name: The tool that was called.
        success: Whether the execution completed without error.
        output: The tool's return value (on success).
        error: Error message (on failure).
    """

    tool_name: str
    success: bool
    output: Any = None
    error: str | None = None


class ToolExecutor:
    """Dispatches tool calls to the registry and wraps results in ``Observation``.

    Guarantees:
        - Never raises — all exceptions are caught and returned as
          ``Observation(success=False, error=...)``.
        - Unknown tools produce an ``Observation`` with an appropriate error.

    Usage::

        executor = ToolExecutor(registry)
        obs = executor.execute("get_weather", {"city": "Beijing"})
        if obs.success:
            print(obs.output)
        else:
            print(obs.error)
    """

    def __init__(
        self,
        registry: ToolRegistry,
        permission_policy: PermissionPolicy | None = None,
        audit_logger: AuditLogger | None = None,
    ) -> None:
        self._registry = registry
        self.permission_policy = permission_policy or PermissionPolicy()
        self.audit_logger = audit_logger

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Observation:
        """Execute a tool by name, returning a structured ``Observation``.

        This method **never raises**.  All errors are captured and returned
        inside the ``Observation``.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Keyword arguments for the tool.

        Returns:
            An ``Observation`` with either ``output`` or ``error`` populated.
        """
        # 1. Look up tool
        tool = self._registry.get(tool_name)
        if tool is None:
            if self.audit_logger is not None:
                self.audit_logger.log(
                    tool_name,
                    arguments,
                    allowed=False,
                    success=False,
                    error=f"Tool '{tool_name}' is not registered.",
                )
            return Observation(
                tool_name=tool_name,
                success=False,
                error=f"Tool '{tool_name}' is not registered.",
            )

        decision = self.permission_policy.check(tool_name, arguments)
        if not decision.allowed:
            if self.audit_logger is not None:
                self.audit_logger.log(
                    tool_name,
                    arguments,
                    allowed=False,
                    success=False,
                    error=decision.reason,
                )
            return Observation(
                tool_name=tool_name,
                success=False,
                error=decision.reason,
            )

        # 2. Execute
        try:
            result = tool.run(**arguments)
            if self.audit_logger is not None:
                self.audit_logger.log(
                    tool_name,
                    arguments,
                    allowed=True,
                    success=True,
                    output=result,
                )
            return Observation(
                tool_name=tool_name,
                success=True,
                output=result,
            )
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            error = f"{type(exc).__name__}: {exc}"
            if self.audit_logger is not None:
                self.audit_logger.log(
                    tool_name,
                    arguments,
                    allowed=True,
                    success=False,
                    error=error,
                )
            return Observation(
                tool_name=tool_name,
                success=False,
                error=error,
            )

    def execute_tool_call(self, tool_call: Any) -> Observation:
        """Convenience: accept a ``ToolCall`` pydantic model directly.

        Args:
            tool_call: A ``miniclaw.agent.state.ToolCall`` instance.

        Returns:
            An ``Observation``.
        """
        return self.execute(tool_call.tool_name, tool_call.arguments)
