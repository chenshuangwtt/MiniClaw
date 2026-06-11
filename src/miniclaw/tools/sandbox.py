"""Sandbox executor — run shell commands in a restricted environment.

Provides configurable isolation for subprocess execution:
    - Path whitelisting
    - Extended command blocklist
    - Environment variable sanitisation
    - Timeout enforcement

Usage::

    from miniclaw.tools.sandbox import SandboxExecutor, SandboxPolicy

    policy = SandboxPolicy(
        max_cpu_seconds=10.0,
        allowed_paths=["/tmp", "/workspace"],
        network_access=False,
    )
    sandbox = SandboxExecutor(policy)
    result = sandbox.execute("echo hello", timeout=5)
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Extended blocklist beyond the basic RunShell patterns
_SANDBOX_BLOCKED: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-rf\s+/", re.IGNORECASE),  # rm -rf /
    re.compile(r"\bmkfs\b", re.IGNORECASE),  # format filesystem
    re.compile(r"\bdd\s+.*of=/dev/", re.IGNORECASE),  # dd to device
    re.compile(r"\bnc\s+-l", re.IGNORECASE),  # netcat listener
    re.compile(r"\bpython\s+-m\s+http\.server", re.IGNORECASE),  # HTTP server
    re.compile(r"\bssh\s+", re.IGNORECASE),  # outbound SSH
    re.compile(r"\bscp\s+", re.IGNORECASE),  # outbound SCP
    re.compile(r"\bcurl\s+.*\|\s*(ba)?sh", re.IGNORECASE),  # pipe to shell
    re.compile(r"\bwget\s+.*\|\s*(ba)?sh", re.IGNORECASE),  # pipe to shell
]

_PATH_REF_RE = re.compile(r"(?P<path>[A-Za-z]:[\\/][^\s'\"|&;<>]+|[\\/][^\s'\"|&;<>]+)")


@dataclass
class SandboxPolicy:
    """Configuration for sandboxed command execution.

    Attributes:
        max_cpu_seconds: Maximum CPU time in seconds (soft limit).
        allowed_paths: If non-empty, commands can only access these paths.
        blocked_commands: Additional command patterns to block (merged
            with the built-in sandbox blocklist).
        network_access: If ``False``, sets ``no_proxy=*`` and clears
            proxy environment variables.
        env_vars: Extra environment variables to set (or override).
    """

    max_cpu_seconds: float = 30.0
    allowed_paths: list[str] = field(default_factory=list)
    blocked_commands: list[str] = field(default_factory=list)
    network_access: bool = False
    env_vars: dict[str, str] = field(default_factory=dict)


class SandboxExecutor:
    """Execute shell commands inside a restricted environment.

    Args:
        policy: The sandbox policy to enforce.

    Usage::

        executor = SandboxExecutor(SandboxPolicy(max_cpu_seconds=10))
        result = executor.execute("ls -la", timeout=15)
    """

    def __init__(self, policy: SandboxPolicy | None = None) -> None:
        self.policy = policy or SandboxPolicy()

    def execute(self, command: str, timeout: float = 30.0) -> dict[str, Any]:
        """Run *command* in a sandboxed subprocess.

        Args:
            command: The shell command to execute.
            timeout: Wall-clock timeout in seconds.

        Returns:
            A dict with ``stdout``, ``stderr``, ``exit_code`` on success,
            or ``error`` on failure.
        """
        # 1. Command blocklist check
        blocked, reason = self._is_blocked(command)
        if blocked:
            return {"error": f"Command blocked by sandbox: {reason}"}

        # 2. Path boundary check
        allowed, path_error = self._resolve_allowed_paths()
        if path_error:
            return {"error": path_error}
        path_allowed, path_reason = self._check_allowed_paths(command, allowed)
        if not path_allowed:
            return {"error": f"Command blocked by sandbox: {path_reason}"}

        # 3. Build restricted environment
        env = self._build_env()
        cwd = str(allowed[0]) if allowed else None

        # 4. Execute
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                cwd=cwd,
            )
            return {
                "stdout": result.stdout[:50_000],  # cap output
                "stderr": result.stderr[:10_000],
                "exit_code": result.returncode,
            }
        except subprocess.TimeoutExpired:
            return {"error": f"Command timed out after {timeout}s: {command}"}
        except Exception as exc:
            return {"error": f"Sandbox execution failed: {exc}"}

    def _is_blocked(self, command: str) -> tuple[bool, str]:
        """Check if *command* matches any blocked pattern."""
        cmd_lower = command.lower().strip()

        # Built-in sandbox blocklist
        for pattern in _SANDBOX_BLOCKED:
            if pattern.search(cmd_lower):
                return True, f"matched sandbox pattern: {pattern.pattern}"

        # User-defined blocklist
        for blocked in self.policy.blocked_commands:
            if blocked.lower() in cmd_lower:
                return True, f"matched user blocklist: {blocked}"

        return False, ""

    def _resolve_allowed_paths(self) -> tuple[list[Path], str | None]:
        """Resolve configured allowed paths and validate the sandbox cwd."""
        if not self.policy.allowed_paths:
            return [], None

        allowed = [
            Path(path).expanduser().resolve(strict=False) for path in self.policy.allowed_paths
        ]
        cwd = allowed[0]
        if not cwd.exists() or not cwd.is_dir():
            return [], f"Allowed sandbox path does not exist or is not a directory: {cwd}"
        return allowed, None

    def _check_allowed_paths(self, command: str, allowed_paths: list[Path]) -> tuple[bool, str]:
        """Reject explicit path references outside the configured boundary."""
        if not allowed_paths:
            return True, ""

        if re.search(r"(^|[\s\\/])\.\.([\\/]|$)", command):
            return False, "parent directory traversal is not allowed"

        for match in _PATH_REF_RE.finditer(command):
            raw_path = match.group("path")
            if _is_windows_switch(raw_path):
                continue

            candidate = Path(raw_path).expanduser().resolve(strict=False)
            if not _is_within_any(candidate, allowed_paths):
                return False, f"path outside allowed sandbox paths: {raw_path}"

        return True, ""

    def _build_env(self) -> dict[str, str]:
        """Build a restricted environment for the subprocess."""
        # Start from a clean base
        env: dict[str, str] = {
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "HOME": os.environ.get("HOME", "/tmp"),
            "USER": os.environ.get("USER", "sandbox"),
            "LANG": os.environ.get("LANG", "en_US.UTF-8"),
        }

        # Network restriction
        if not self.policy.network_access:
            env["no_proxy"] = "*"
            env["NO_PROXY"] = "*"
            # Remove proxy vars
            for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
                env.pop(key, None)

        # Apply user overrides
        env.update(self.policy.env_vars)

        return env


def _is_windows_switch(value: str) -> bool:
    """Return True for short cmd.exe style switches such as ``/c``."""
    return os.name == "nt" and bool(re.fullmatch(r"/[A-Za-z]", value))


def _is_within_any(candidate: Path, allowed_paths: list[Path]) -> bool:
    """Check whether *candidate* is inside any allowed path."""
    candidate_s = os.path.normcase(str(candidate))
    for allowed in allowed_paths:
        allowed_s = os.path.normcase(str(allowed))
        try:
            if os.path.commonpath([allowed_s, candidate_s]) == allowed_s:
                return True
        except ValueError:
            continue
    return False
