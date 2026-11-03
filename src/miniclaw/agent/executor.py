"""ToolExecutor — safe dispatch of tool calls with structured observations."""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel

from miniclaw.tools.audit import AuditLogger
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.timeout import CancellationToken, TimedOutError, run_with_timeout

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
    error_kind: str | None = None
    retryable: bool = False


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
        default_timeout: float | None = None,
    ) -> None:
        self._registry = registry
        self.permission_policy = permission_policy or PermissionPolicy()
        self.audit_logger = audit_logger
        self.default_timeout = default_timeout
        self._active_tokens: dict[str, CancellationToken] = {}

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> Observation:
        """Execute a tool by name, returning a structured ``Observation``.

        This method **never raises**.  All errors are captured and returned
        inside the ``Observation``.

        If the tool has a ``timeout`` attribute (or ``default_timeout`` is set),
        the execution is wrapped with a wall-clock timeout.

        Args:
            tool_name: Name of the tool to invoke.
            arguments: Keyword arguments for the tool.

        Returns:
            An ``Observation`` with either ``output`` or ``error`` populated.
        """
        # 1. Look up tool
        tool = self._registry.get(tool_name)
        if tool is None:
            return self._failed_observation(
                tool_name,
                arguments,
                f"Tool '{tool_name}' does not exist / is not registered.",
                "unknown_tool",
            )

        validation_error = _validate_arguments(tool.schema, arguments)
        if validation_error:
            return self._failed_observation(tool_name, arguments, validation_error, "validation")

        decision = self.permission_policy.check(tool_name, arguments, tool.capability)
        if not decision.allowed:
            return self._failed_observation(tool_name, arguments, decision.reason, "permission")

        # 2. Execute (with optional timeout)
        timeout = getattr(tool, "timeout", None) or self.default_timeout
        token = CancellationToken()
        self._active_tokens[tool_name] = token

        try:
            if timeout is not None:
                result = run_with_timeout(tool.run, kwargs=arguments, timeout=timeout, token=token)
            else:
                result = tool.run(**arguments)

            # Built-in adapters use an error payload for transport-level failures.
            # Treat it as a failed observation rather than letting a later final
            # answer incorrectly turn the overall run into a success.
            if isinstance(result, dict) and result.get("error"):
                return self._failed_observation(
                    tool_name,
                    arguments,
                    str(result["error"]),
                    "execution",
                    retryable=bool(result.get("retryable", False)),
                )
            if isinstance(result, dict) and result.get("exit_code", 0) != 0:
                return self._failed_observation(
                    tool_name,
                    arguments,
                    f"Tool exited with status {result['exit_code']}: {result.get('stderr', '')}",
                    "execution",
                )

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
        except TimedOutError:
            error = f"Tool '{tool_name}' timed out after {timeout}s"
            logger.warning(error)
            return self._failed_observation(tool_name, arguments, error, "timeout", retryable=True)
        except Exception as exc:
            logger.exception("Tool '%s' raised an exception", tool_name)
            error = f"{type(exc).__name__}: {exc}"
            return self._failed_observation(tool_name, arguments, error, "execution")
        finally:
            self._active_tokens.pop(tool_name, None)

    def cancel_execution(self, tool_name: str) -> bool:
        """Request cancellation of a running tool.

        Args:
            tool_name: Name of the tool to cancel.

        Returns:
            ``True`` if a cancellation token was found and cancelled.
        """
        token = self._active_tokens.get(tool_name)
        if token is not None:
            token.cancel()
            return True
        return False

    def _failed_observation(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        error: str,
        error_kind: str,
        retryable: bool = False,
    ) -> Observation:
        if self.audit_logger is not None:
            self.audit_logger.log(
                tool_name,
                arguments,
                allowed=error_kind != "permission",
                success=False,
                error=error,
            )
        return Observation(
            tool_name=tool_name,
            success=False,
            error=error,
            error_kind=error_kind,
            retryable=retryable,
        )

    def execute_tool_call(self, tool_call: Any) -> Observation:
        """Convenience: accept a ``ToolCall`` pydantic model directly.

        Args:
            tool_call: A ``miniclaw.agent.state.ToolCall`` instance.

        Returns:
            An ``Observation``.
        """
        return self.execute(tool_call.tool_name, tool_call.arguments)


def _validate_arguments(schema: dict[str, Any], arguments: dict[str, Any]) -> str | None:
    """Validate the practical JSON Schema subset used by MiniClaw tools."""
    if not isinstance(arguments, dict):
        return "Tool arguments must be a JSON object."
    required = schema.get("required", [])
    for name in required:
        if name not in arguments:
            return f"Missing required argument: {name}."
    for name, value in arguments.items():
        property_schema = schema.get("properties", {}).get(name)
        if property_schema is None:
            if schema.get("additionalProperties") is False:
                return f"Unexpected argument: {name}."
            continue
        expected = property_schema.get("type")
        if expected and not _matches_json_type(value, expected):
            return f"Argument '{name}' must be {expected}, got {type(value).__name__}."
        choices = property_schema.get("enum")
        if choices is not None and value not in choices:
            return f"Argument '{name}' must be one of {choices}."
    return None


def _matches_json_type(value: Any, expected: str) -> bool:
    if expected == "string":
        return isinstance(value, str)
    if expected == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if expected == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if expected == "boolean":
        return isinstance(value, bool)
    if expected == "array":
        return isinstance(value, list)
    if expected == "object":
        return isinstance(value, dict)
    if expected == "null":
        return value is None
    return True
