"""Demo 1: 项目结构分析 — 展示多工具编排能力。

运行: python examples/demo_project_analysis.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniclaw.agent.loop import AgentLoop
from miniclaw.llm.fake import FakeLLM
from miniclaw.tools.file_tools import ListFiles, ReadFile, WriteFile
from miniclaw.tools.registry import ToolRegistry
from miniclaw.tools.shell_tool import RunShell

# --- 构建 FakeLLM ---
llm = FakeLLM([
    json.dumps({
        "type": "tool_call",
        "thought": "先看看项目根目录有什么文件。",
        "tool_name": "list_files",
        "arguments": {"path": "."}
    }),
    json.dumps({
        "type": "tool_call",
        "thought": "根目录有 src/ 和 tests/，先看 src/miniclaw 的结构。",
        "tool_name": "list_files",
        "arguments": {"path": "src/miniclaw"}
    }),
    json.dumps({
        "type": "tool_call",
        "thought": "看到 agent/ 子目录，读取 loop.py 了解核心逻辑。",
        "tool_name": "read_file",
        "arguments": {"path": "src/miniclaw/agent/loop.py"}
    }),
    json.dumps({
        "type": "tool_call",
        "thought": "现在统计所有 Python 文件的代码行数。",
        "tool_name": "run_shell",
        "arguments": {"command": "find . -name '*.py' -not -path './.venv/*' | head -20"}
    }),
    json.dumps({
        "type": "final_answer",
        "thought": "我已经收集了足够的项目结构信息。",
        "answer": (
            "## 项目结构分析\n\n"
            "**MiniClaw** 是一个轻量级 Agent Harness，包含以下模块：\n\n"
            "- `src/miniclaw/agent/` — Agent 核心（loop, parser, executor, recovery, context, trace）\n"
            "- `src/miniclaw/tools/` — 工具系统（base, registry, file_tools, shell_tool）\n"
            "- `src/miniclaw/llm/` — LLM 抽象层（base, fake, openai_client）\n"
            "- `src/miniclaw/storage/` — SQLite 持久化\n"
            "- `tests/` — 16 个测试文件，327 个测试用例\n\n"
            "项目结构清晰，模块职责单一，依赖方向明确。"
        )
    }),
])

# --- 注册工具 ---
registry = ToolRegistry()
registry.register(ListFiles())
registry.register(ReadFile())
registry.register(WriteFile())
registry.register(RunShell())

# --- 运行 ---
print("🐾 MiniClaw — Demo 1: 项目结构分析")
print("=" * 50)
task = "分析当前目录的项目结构，列出所有 Python 文件，并统计代码行数。"
print(f"📋 Task: {task}\n")

agent = AgentLoop(llm=llm, registry=registry, max_steps=10)
result = agent.run(task)

for s in result.trace.steps:
    if s.parsed_action == "tool_call":
        obs = str(s.observation or s.error or "")[:120]
        print(f"  Step {s.step}: 🔧 {s.tool_name}({s.arguments})")
        print(f"         → {obs}")
    elif s.parsed_action == "final_answer":
        print(f"  Step {s.step}: ✅ final_answer")
    else:
        print(f"  Step {s.step}: {s.parsed_action}")

print(f"\n{'=' * 50}")
if result.success:
    print(f"✅ Answer ({result.steps_taken} steps):\n")
    print(result.answer)
else:
    print(f"❌ Failed: {result.error}")
