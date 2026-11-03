"""Offline repository-maintenance agent demo.

The demo copies an intentionally broken fixture to a temporary workspace. A
scripted ``FakeLLM`` inspects it, requests write and shell permissions, fixes
the source file, then runs the fixture's tests. No API key or repository data
is required, and the checked-in fixture is never modified.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.fake import FakeLLM
from miniclaw.tools.file_tools import ListFiles, ReadFile, WriteFile
from miniclaw.tools.permissions import PermissionPolicy
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.sandbox import SandboxExecutor, SandboxPolicy
from miniclaw.tools.shell_tool import RunShell


ROOT = Path(__file__).resolve().parent
FIXTURE = ROOT / "repository_maintenance_fixture"


def main() -> None:
    """Run the isolated offline workflow and print its durable trace."""
    with tempfile.TemporaryDirectory(prefix="miniclaw-repair-") as directory:
        workspace = Path(directory)
        shutil.copytree(FIXTURE, workspace / "fixture")

        registry = ToolRegistry()
        registry.register(ListFiles(workspace_root=workspace))
        registry.register(ReadFile(workspace_root=workspace))
        registry.register(WriteFile(workspace_root=workspace, allow_write=True))
        registry.register(
            RunShell(
                allow_shell=True,
                allowed_prefixes=["python -m pytest"],
                sandbox=SandboxExecutor(SandboxPolicy(allowed_paths=[str(workspace)])),
            )
        )
        fixed_source = (
            '"""Repaired by the offline MiniClaw demo."""\n\n\n'
            "def add(left: int, right: int) -> int:\n"
            '    """Return the sum of two integers."""\n'
            "    return left + right\n"
        )
        llm = FakeLLM(
            [
                '{"type":"tool_call","tool_name":"list_files","arguments":{"path":"fixture"}}',
                '{"type":"tool_call","tool_name":"read_file","arguments":{"path":"fixture/calculator.py"}}',
                '{"type":"tool_call","tool_name":"read_file","arguments":{"path":"fixture/test_calculator.py"}}',
                '{"type":"tool_call","tool_name":"write_file","arguments":{"path":"fixture/calculator.py","content":'
                + __import__("json").dumps(fixed_source, ensure_ascii=False)
                + "}}",
                '{"type":"tool_call","tool_name":"run_shell","arguments":{"command":"python -m pytest -q fixture"}}',
                '{"type":"final_answer","answer":"Repaired calculator.add and verified the fixture test passes."}',
            ]
        )
        policy = PermissionPolicy(
            allow_file_write=True,
            allow_shell=True,
            shell_allowed_prefixes=["python -m pytest"],
        )
        result = AgentLoop(llm=llm, registry=registry, permission_policy=policy).run(
            "Inspect the fixture, repair its implementation, and verify it with tests."
        )
        print(f"success={result.success} steps={result.steps_taken}")
        print(result.answer or result.error)
        for step in result.trace.steps:
            print(f"[{step.step}] {step.parsed_action} {step.tool_name or ''} {step.error or ''}")
        if not result.success:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
