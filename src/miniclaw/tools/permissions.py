"""Permission policy for tool execution."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable

from miniclaw.tools.base import ToolCapability


ApprovalCallback = Callable[[str, dict[str, Any]], bool]
SHELL_CONTROL_OPERATORS = ("&&", "||", ";", "|", "&", "\n", "\r")


def command_matches_allowed_prefix(command: str, prefixes: list[str]) -> bool:
    """Return whether a shell command matches an allowlisted command prefix."""
    normalized = command.strip().lower()
    if not normalized:
        return False
    if any(operator in normalized for operator in SHELL_CONTROL_OPERATORS):
        return False

    for prefix in prefixes:
        normalized_prefix = prefix.strip().lower()
        if not normalized_prefix:
            continue
        if normalized == normalized_prefix or normalized.startswith(f"{normalized_prefix} "):
            return True
    return False


@dataclass(frozen=True)
class PermissionDecision:
    """Result of checking whether a tool call is allowed."""

    allowed: bool
    reason: str = ""


@dataclass
class PermissionPolicy:
    """Central policy object for tool permissions.

    The policy is intentionally small: it gates sensitive built-in tools and
    supports an optional human approval callback for future interactive flows.
    """

    allow_file_write: bool = False  # Off by default — explicit opt-in required
    allow_shell: bool = False  # Off by default — explicit opt-in required
    allow_search: bool = False  # Off by default — explicit opt-in required
    allow_network: bool = False
    allow_external_side_effects: bool = False
    shell_allowed_prefixes: list[str] | None = None
    approval_required_tools: set[str] = field(default_factory=set)
    approval_callback: ApprovalCallback | None = None

    def check(
        self,
        tool_name: str,
        arguments: dict[str, Any],
        capability: ToolCapability | None = None,
    ) -> PermissionDecision:
        """Return whether *tool_name* may run with *arguments*."""
        capability = capability or _legacy_capability(tool_name)
        if capability is ToolCapability.FILE_WRITE and not self.allow_file_write:
            return PermissionDecision(False, "File writes are disabled by permission policy.")

        if capability is ToolCapability.SHELL_EXECUTION:
            if not self.allow_shell:
                return PermissionDecision(
                    False, "Shell execution is disabled by permission policy."
                )
            if self.shell_allowed_prefixes is not None:
                command = str(arguments.get("command", ""))
                allowed = command_matches_allowed_prefix(command, self.shell_allowed_prefixes)
                if not allowed:
                    return PermissionDecision(
                        False,
                        f"Shell command is not in allowed prefixes: {self.shell_allowed_prefixes}",
                    )

        if (
            capability is ToolCapability.NETWORK_ACCESS
            and not (self.allow_network or (tool_name == "web_search" and self.allow_search))
            and tool_name not in self.approval_required_tools
        ):
            return PermissionDecision(False, "Network access is disabled by permission policy.")

        if (
            capability is ToolCapability.EXTERNAL_SIDE_EFFECT
            and not self.allow_external_side_effects
        ):
            return PermissionDecision(
                False, "External side effects are disabled by permission policy."
            )

        if tool_name in self.approval_required_tools:
            if self.approval_callback is None:
                return PermissionDecision(False, f"Tool '{tool_name}' requires approval.")
            if not self.approval_callback(tool_name, arguments):
                return PermissionDecision(
                    False, f"Tool '{tool_name}' was rejected by approval hook."
                )

        return PermissionDecision(True)


def _legacy_capability(tool_name: str) -> ToolCapability:
    """Keep direct policy calls compatible with older tool-name callers."""
    return {
        "write_file": ToolCapability.FILE_WRITE,
        "run_shell": ToolCapability.SHELL_EXECUTION,
        "web_search": ToolCapability.NETWORK_ACCESS,
    }.get(tool_name, ToolCapability.READ_ONLY)
