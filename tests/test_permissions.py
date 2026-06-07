"""Tests for tools/permissions.py."""

from miniclaw.tools.permissions import PermissionPolicy


class TestPermissionPolicy:
    def test_allows_safe_default(self):
        decision = PermissionPolicy().check("read_file", {"path": "README.md"})
        assert decision.allowed is True

    def test_blocks_file_write(self):
        decision = PermissionPolicy(allow_file_write=False).check(
            "write_file", {"path": "x.txt", "content": "hi"}
        )
        assert decision.allowed is False
        assert "File writes" in decision.reason

    def test_blocks_shell(self):
        decision = PermissionPolicy(allow_shell=False).check("run_shell", {"command": "echo hi"})
        assert decision.allowed is False
        assert "Shell execution" in decision.reason

    def test_shell_prefixes_allow_matching_command(self):
        decision = PermissionPolicy(shell_allowed_prefixes=["echo"]).check(
            "run_shell", {"command": "echo hi"}
        )
        assert decision.allowed is True

    def test_shell_prefixes_block_unmatched_command(self):
        decision = PermissionPolicy(shell_allowed_prefixes=["echo"]).check(
            "run_shell", {"command": "python --version"}
        )
        assert decision.allowed is False
        assert "allowed prefixes" in decision.reason

    def test_shell_prefixes_require_command_boundary(self):
        decision = PermissionPolicy(shell_allowed_prefixes=["echo"]).check(
            "run_shell", {"command": "echoevil hello"}
        )
        assert decision.allowed is False
        assert "allowed prefixes" in decision.reason

    def test_shell_prefixes_block_command_chaining(self):
        decision = PermissionPolicy(shell_allowed_prefixes=["echo"]).check(
            "run_shell", {"command": "echo hello && python --version"}
        )
        assert decision.allowed is False
        assert "allowed prefixes" in decision.reason

    def test_approval_callback_allows(self):
        policy = PermissionPolicy(
            approval_required_tools={"write_file"},
            approval_callback=lambda tool, args: True,
        )
        decision = policy.check("write_file", {"path": "x.txt", "content": "hi"})
        assert decision.allowed is True

    def test_approval_callback_rejects(self):
        policy = PermissionPolicy(
            approval_required_tools={"write_file"},
            approval_callback=lambda tool, args: False,
        )
        decision = policy.check("write_file", {"path": "x.txt", "content": "hi"})
        assert decision.allowed is False
        assert "rejected" in decision.reason
