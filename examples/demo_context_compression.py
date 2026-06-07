"""Demo 3: 上下文压缩 — 展示 ContextManager 自动压缩能力。

运行: python examples/demo_context_compression.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniclaw.agent.context import ContextManager

# --- 创建一个 token 预算很小的 ContextManager（便于演示） ---
ctx = ContextManager(max_context_tokens=50, recent_keep=4)

print("🐾 MiniClaw — Demo 3: 上下文压缩")
print("=" * 50)
print(f"Token 预算: {ctx.max_context_tokens}")
print(f"保留最近: {ctx.recent_keep} 条消息\n")

# --- 模拟长对话：15 轮 user + assistant + tool ---
print("正在模拟 15 轮对话...")
for i in range(1, 16):
    ctx.add_message("user", f"第 {i} 个问题：请帮我分析项目中 module_{i} 的代码结构和实现细节。")
    ctx.add_message("assistant", f"好的，我来分析 module_{i}。")
    ctx.add_observation(
        "read_file",
        {"path": f"src/module_{i}.py"},
        f"这是 module_{i} 的完整代码，包含了 {i * 100} 行 Python 代码，主要实现了...",
    )

print(f"\n--- 压缩前 ---")
print(f"消息数: {ctx.message_count}")
print(f"估算 tokens: {ctx.estimate_tokens()}")
print(f"需要压缩: {ctx.should_compress()}")

# --- 触发压缩 ---
compressed = ctx.compress()

print(f"\n--- 压缩后 ---")
print(f"消息数: {ctx.message_count}")
print(f"估算 tokens: {ctx.estimate_tokens()}")
print(f"是否执行了压缩: {compressed}")

# --- 展示压缩后的消息结构 ---
print(f"\n压缩后的消息列表:")
for i, msg in enumerate(ctx.get_messages()):
    role = msg["role"]
    content = msg["content"]
    # 截断显示
    if len(content) > 120:
        content = content[:120] + "..."
    print(f"  [{i}] {role}: {content}")

# --- 展示 build_messages 输出 ---
print(f"\n--- build_messages 输出 ---")
messages = ctx.build_messages(
    system_prompt="你是一个代码分析助手。",
    tools_prompt="## Tools\n- read_file: 读取文件",
    current_task="请继续分析。",
)
for i, msg in enumerate(messages):
    role = msg["role"]
    content = msg["content"]
    if len(content) > 80:
        content = content[:80] + "..."
    print(f"  [{i}] {role}: {content}")

print(f"\n{'=' * 50}")
print("✅ 压缩演示完成。")
print("默认摘要器是规则提取（零开销），可替换为 LLM 摘要器：")
print('  ctx = ContextManager(summarizer=lambda msgs: llm.generate("总结: " + str(msgs)))')
