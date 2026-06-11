"""Shell execution tool with safety guardrails."""

from __future__ import annotations

import re
import subprocess
from typing import TYPE_CHECKING, Any

from miniclaw.tools.base import Tool
from miniclaw.tools.permissions import command_matches_allowed_prefix

if TYPE_CHECKING:
    from miniclaw.tools.sandbox import SandboxExecutor

# Default timeout in seconds.
DEFAULT_TIMEOUT = 30

# Commands (or command prefixes) that are blocked for safety.
BLOCKED_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\b"),
    re.compile(r"\bsudo\b"),
    re.compile(r"\bchmod\b"),
    re.compile(r"\bchown\b"),
    re.compile(r"\bkill\b"),
    re.compile(r"\bmkfs\b"),
    re.compile(r"\bdd\b"),
    re.compile(r"\bformat\b"),
    re.compile(r"\bcurl\b.*\|\s*bash"),
    re.compile(r"\bwget\b.*\|\s*bash"),
    re.compile(r"\bcurl\b.*\|\s*sh"),
    re.compile(r"\bwget\b.*\|\s*sh"),
]


def is_command_safe(command: str) -> tuple[bool, str]:
    """Check whether *command* passes the safety filter.

    Returns:
        ``(True, "")`` if safe, ``(False, reason)`` if blocked.
    """
    normalized = command.strip().lower()
    for pattern in BLOCKED_PATTERNS:
        if pattern.search(normalized):
            return False, f"Blocked by safety filter: pattern '{pattern.pattern}' matched."
    return True, ""


class RunShell(Tool):
    """Execute a shell command with safety guardrails."""

    name = "run_shell"
    description = (
        "Execute a shell command and return stdout, stderr, and exit code. "
        "Dangerous commands (rm, sudo, chmod, chown, kill, curl|bash, etc.) are blocked."
    )
    schema = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "Shell command to execute.",
            },
            "timeout": {
                "type": "integer",
                "description": f"Timeout in seconds (default: {DEFAULT_TIMEOUT}).",
                "default": DEFAULT_TIMEOUT,
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        allow_shell: bool = True,
        allowed_prefixes: list[str] | None = None,
        sandbox: SandboxExecutor | None = None,
    ) -> None:
        self.allow_shell = allow_shell
        self.allowed_prefixes = allowed_prefixes
        self.sandbox = sandbox

    def run(self, command: str, timeout: int = DEFAULT_TIMEOUT, **kwargs: Any) -> dict[str, Any]:
        """Execute *command* in a subprocess.

        If a ``sandbox`` was provided, delegates to the sandbox executor
        for restricted execution.

        Returns:
            A dict with ``stdout``, ``stderr``, ``exit_code``,
            or ``error`` if the command was blocked or timed out.
        """
        if not self.allow_shell:
            return {"error": "Shell execution is disabled by tool permissions."}

        if self.allowed_prefixes is not None:
            if not command_matches_allowed_prefix(command, self.allowed_prefixes):
                return {"error": f"Command is not in the allowed prefixes: {self.allowed_prefixes}"}

        safe, reason = is_command_safe(command)
        if not safe:
            return {"error": reason}

        # Delegate to sandbox if available
        if self.sandbox is not None:
            return self.sandbox.execute(command, timeout=timeout)

        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            return {
                "stdout": result.stdout,
                "stderr": result.stderr,
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s: {command}"}
        except Exception as exc:
            return {"error": f"Failed to execute command: {exc}"}
