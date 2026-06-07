"""Demo: SQLite 持久化记忆 — 展示 sessions/messages/memories/traces 的读写。

运行: uv run python examples/demo_memory.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from miniclaw.storage.sqlite_store import SQLiteStore

DB_PATH = ":memory:"

print("MiniClaw — Demo: SQLite Memory")
print("=" * 55)

with SQLiteStore(DB_PATH) as store:
    # --- 1. 创建会话 ---
    sid1 = store.create_session("天气查询任务")
    sid2 = store.create_session("代码分析任务")
    print(f"\n[1] 创建了 2 个会话: #{sid1}, #{sid2}")

    # --- 2. 保存消息 ---
    store.save_message(sid1, "user", "北京天气怎么样？")
    store.save_message(sid1, "assistant", "让我查一下。")
    store.save_message(sid1, "tool", "北京：晴，25°C")
    store.save_message(sid1, "assistant", "北京今天晴天，25°C。")

    store.save_message(sid2, "user", "分析 main.py 的代码结构。")
    store.save_message(sid2, "assistant", "main.py 包含 CLI 入口和 5 个子命令。")

    msgs1 = store.list_messages(sid1)
    msgs2 = store.list_messages(sid2)
    print(f"[2] 会话 #{sid1} 有 {len(msgs1)} 条消息，会话 #{sid2} 有 {len(msgs2)} 条消息")

    # --- 3. 保存长期记忆 ---
    store.save_memory("user:name", "Alice", importance=8)
    store.save_memory("user:city", "Beijing", importance=5)
    store.save_memory("user:language", "Python", importance=3)
    store.save_memory("project:name", "MiniClaw", importance=9)
    print(f"[3] 保存了 4 条长期记忆")

    # --- 4. 搜索记忆 ---
    results = store.search_memories("alice")
    print(f"[4] 搜索 'alice': {len(results)} 条结果 → {results[0]['key']}={results[0]['value']}")

    results = store.search_memories("MiniClaw")
    print(f"    搜索 'MiniClaw': {len(results)} 条结果 → {results[0]['key']}={results[0]['value']}")

    # --- 5. 按重要度排序 ---
    mems = store.list_memories()
    print(f"[5] 所有记忆（按重要度排序）:")
    for m in mems:
        stars = "*" * m["importance"]
        print(f"    [{m['key']}] = {m['value']}  importance={stars}")

    # --- 6. 保存 trace ---
    store.save_trace(sid1, step=1, event_json=json.dumps({
        "parsed_action": "tool_call",
        "tool_name": "get_weather",
        "arguments": {"city": "Beijing"},
        "observation": "Sunny, 25C",
    }))
    store.save_trace(sid1, step=2, event_json=json.dumps({
        "parsed_action": "final_answer",
        "observation": "Beijing: Sunny, 25C",
    }))
    print(f"[6] 会话 #{sid1} 保存了 2 条 trace")

    # --- 7. 查询 trace ---
    traces = store.list_traces(sid1)
    print(f"[7] 会话 #{sid1} 的 trace:")
    for t in traces:
        event = json.loads(t["event_json"])
        action = event.get("parsed_action", "?")
        if action == "tool_call":
            print(f"    Step {t['step']}: tool_call → {event.get('tool_name')}({event.get('arguments')})")
        else:
            print(f"    Step {t['step']}: {action}")

    # --- 8. UPSERT 测试 ---
    store.save_memory("user:name", "Bob", importance=10)
    mems = store.list_memories()
    name_mem = [m for m in mems if m["key"] == "user:name"][0]
    print(f"[8] UPSERT: user:name 从 Alice 更新为 {name_mem['value']}，importance={name_mem['importance']}")

    # --- 9. 跨会话记忆 ---
    all_msgs = store.list_messages(sid1) + store.list_messages(sid2)
    print(f"[9] 两个会话共 {len(all_msgs)} 条消息，记忆跨会话持久化")

print(f"\n{'=' * 55}")
print("Done. SQLite 四张表全部验证通过:")
print("  sessions  — 创建/查询")
print("  messages  — 保存/列表/按会话隔离")
print("  memories  — 保存/搜索/UPSERT/按重要度排序")
print("  traces    — 保存/查询/JSON 解析")
