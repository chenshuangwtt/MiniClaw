"""Demo 2: 异常工具调用恢复 — 展示 RecoveryManager 自纠正能力。

运行: python examples/demo_recovery.py
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

# --- 构建 FakeLLM（故意第一次用错工具名） ---
llm = FakeLLM([
    # Step 1: 故意使用不存在的工具名
    json.dumps({
        "type": "tool_call",
        "thought": "我需要读取文件内容。",
        "tool_name": "open_file",       # ← 不存在！
        "arguments": {"path": "README.md"}
    }),
    # Step 2: 收到恢复提示后，修正为正确的工具名
    json.dumps({
        "type": "tool_call",
        "thought": "抱歉，工具名应该是 read_file，我来修正。",
        "tool_name": "read_file",       # ← 修正后
        "arguments": {"path": "README.md"}
    }),
    # Step 3: 最终回答
    json.dumps({
        "type": "final_answer",
        "thought": "文件读取成功。",
        "answer": "成功读取 README.md。项目是一个从零实现的轻量级 Agent Harness，不依赖 LangChain/AutoGen/CrewAI。"
    }),
])

# --- 注册工具 ---
registry = ToolRegistry()
registry.register(ListFiles())
registry.register(ReadFile())
registry.register(WriteFile())
registry.register(RunShell())

# --- 运行 ---
print("🐾 MiniClaw — Demo 2: 异常工具调用恢复")
print("=" * 50)
task = "读取 README.md 文件的内容。"
print(f"📋 Task: {task}\n")

agent = AgentLoop(llm=llm, registry=registry, max_steps=10)
result = agent.run(task)

for s in result.trace.steps:
    if s.parsed_action == "tool_call":
        if s.error:
            print(f"  Step {s.step}: 🔧 {s.tool_name}({s.arguments})")
            print(f"         ❌ ERROR: {s.error[:100]}")
        else:
            obs = str(s.observation or "")[:100]
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
