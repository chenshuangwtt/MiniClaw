# MiniClaw 代码解读：Agent Loop、工具调用与恢复机制

本文档包含两个部分：

- **Part A** — 真实运行路径：`uv run python main.py run -v "列出当前目录并总结项目结构"` 的 4 步执行过程。`main.py` 是兼容包装，实际 CLI 实现在 `src/miniclaw/cli.py`。
- **Part B** — Recovery Demo 路径：FakeLLM 模拟 `open_file` 错误 → 自动修正为 `read_file` 的 5 步恢复过程

---

## 目录

- [Part A：真实运行路径](#part-a真实运行路径)
  - [A.1 入口：main.py → miniclaw.cli](#a1-入口mainpy--miniclawcli)
  - [A.2 AgentLoop.run 启动](#a2-agentlooprun-启动)
  - [A.3 Step 1](#a3-step-1构建-prompt--调用-llm--解析--执行)
  - [A.4 Step 2](#a4-step-2构建带有-step-1-observation-的-prompt--继续探索)
  - [A.5 Step 3](#a5-step-3构建带有前两步-observation-的-prompt--深入-agent-子目录)
  - [A.6 Step 4](#a6-step-4构建带有所有-observation-的-prompt--final_answer-返回)
  - [A.7 完整调用栈](#a7-完整调用栈)
- [Part B：Recovery Demo 路径](#part-brecovery-demo-路径)
  - [B.1 FakeLLM 预设序列（Recovery Demo 专用）](#b1-fakellm-预设序列recovery-demo-专用)
  - [B.2 Step 1：调用不存在的工具](#b2-step-1调用不存在的工具)
  - [B.3 Step 2：RecoveryManager 介入](#b3-step-2recoverymanager-介入)
  - [B.4 Step 3-5：恢复正常执行](#b4-step-3-5恢复正常执行)
- [Part C：模块职责与设计决策](#part-c模块职责与设计决策)
  - [C.1 模块依赖关系图](#c1-模块依赖关系图)
  - [C.2 错误是观察值，不是异常](#c2-错误是观察值不是异常)
  - [C.3 Parser 严格，Recovery 兜底](#c3-parser-严格recovery-兜底)
  - [C.4 ContextManager 压缩时机](#c4-contextmanager-压缩时机)
  - [C.5 error_count 重置逻辑](#c5-error_count-重置逻辑)
  - [C.6 SQLite 四张表关系](#c6-sqlite-四张表关系)

---

## Part A：真实运行路径

以下是对以下命令的真实执行分析：

```bash
uv run python main.py run -v "列出当前目录并总结项目结构"
```

真实输出（4 步，无错误）：

```text
Step 1: 🔧 list_files({'path': '.'})
Step 2: 🔧 list_files({'path': 'src/miniclaw'})
Step 3: 🔧 list_files({'path': 'src/miniclaw/agent'})
Step 4: ✅ final_answer
```

### A.1 入口：main.py → miniclaw.cli

```text
uv run python main.py run -v "列出当前目录并总结项目结构"
  │
  ▼
main.py                             # 兼容包装
  └─ miniclaw.cli.main()            # src/miniclaw/cli.py
       ├─ build_parser()           # 构建 argparse
  ├─ args = parser.parse_args()     # 解析参数：command="run", task="...", verbose=True
  └─ _cmd_run(args, config)         # 创建 LLM、工具注册表、SQLite 会话
```

`_cmd_run` 内部：

```python
def _cmd_run(args, config):
    llm = _create_llm_from_config(config) # → FakeLLM（当前使用的预设序列）
    registry = _register_default_tools() # → ToolRegistry + 4 个工具
    with SQLiteStore(config.storage.db_path) as store:
        agent = AgentLoop(llm, registry) # → 组装 Agent
        result = agent.run(args.task)    # → 进入核心循环
```

### A.2 AgentLoop.run 启动

```text
AgentLoop.run("列出当前目录并总结项目结构")
  │
  ├─ trace = TraceLogger()
  ├─ error_count = 0
  └─ tool_schemas = [list_files, read_file, write_file, run_shell]
```

### A.3 Step 1：构建 Prompt → 调用 LLM → 解析 → 执行

```text
Step 1
  │
  ├─ 构建 prompt
  │   └─ system_prompt + tools_prompt + task="列出当前目录并总结项目结构" + history=[]
  │
  ├─ 调用 LLM
  │   └─ FakeLLM 返回：'{"type": "tool_call", "tool_name": "list_files", "arguments": {"path": "."}}'
  │
  ├─ Parser 解析
  │   ├─ json.loads() → 成功
  │   ├─ type = "tool_call" → 合法
  │   └─ Pydantic 校验 → 通过
  │   └─ return ToolCall(tool_name="list_files", arguments={"path": "."})
  │
  ├─ ToolExecutor 执行 list_files(".")
  │   └─ ListFiles.run(path=".")
  │       └─ Path(".").iterdir() → 扫描目录
  │       └─ return {"path": "...", "entries": [.git, .gitignore, src/, tests/, ...]}
  │
  ├─ Observation 写入上下文
  │   └─ context.add_observation("list_files", {"path": "."}, {"entries": [...]})
  │
  └─ Trace 记录
      └─ trace.log_step(step=1, tool_name="list_files", observation={...})
```

### A.4 Step 2：构建带有 Step 1 observation 的 prompt → 继续探索

```text
Step 2
  │
  ├─ 构建 prompt
  │   └─ system_prompt + tools_prompt + task + history=[
  │       {action: "Called list_files({path: '.'})", observation: {.git, src/, tests/, ...}}
  │     ]
  │
  ├─ 调用 LLM
  │   └─ FakeLLM 返回：'{"type": "tool_call", "tool_name": "list_files", "arguments": {"path": "src/miniclaw"}}'
  │
  ├─ Parser 解析 → ToolCall
  │
  ├─ ToolExecutor 执行 list_files("src/miniclaw")
  │   └─ return {"entries": [agent/, tools/, llm/, storage/, __init__.py, ...]}
  │
  ├─ Observation 写入上下文
  │
  └─ Trace 记录
```

### A.5 Step 3：构建带有前两步 observation 的 prompt → 深入 agent 子目录

```text
Step 3
  │
  ├─ 构建 prompt
  │   └─ system_prompt + tools_prompt + task + history=[
  │       {action: "Called list_files({path: '.'})", observation: {...}},
  │       {action: "Called list_files({path: 'src/miniclaw'})", observation: {agent/, tools/, ...}}
  │     ]
  │
  ├─ 调用 LLM
  │   └─ FakeLLM 返回：'{"type": "tool_call", "tool_name": "list_files", "arguments": {"path": "src/miniclaw/agent"}}'
  │
  ├─ Parser 解析 → ToolCall
  │
  ├─ ToolExecutor 执行 list_files("src/miniclaw/agent")
  │   └─ return {"entries": [loop.py, parser.py, executor.py, recovery.py, context.py, ...]}
  │
  ├─ Observation 写入上下文
  │
  └─ Trace 记录
```

### A.6 Step 4：构建带有所有 observation 的 prompt → final_answer 返回

```text
Step 4
  │
  ├─ 构建 prompt
  │   └─ system_prompt + tools_prompt + task + history=[
  │       {action: "Called list_files({path: '.'})", observation: {.git, src/, tests/, ...}},
  │       {action: "Called list_files({path: 'src/miniclaw'})", observation: {agent/, tools/, ...}},
  │       {action: "Called list_files({path: 'src/miniclaw/agent'})", observation: {loop.py, parser.py, ...}}
  │     ]
  │
  ├─ 调用 LLM
  │   └─ FakeLLM 返回：'{"type": "final_answer", "answer": "## 项目结构分析\\n..."}'
  │
  ├─ Parser 解析 → FinalAnswer
  │
  ├─ AgentLoop 结束
  │   └─ return AgentResult(success=True, answer="## 项目结构分析...", steps_taken=4)
  │
  ├─ 保存 message / trace
  │   ├─ store.save_trace(session_id, step=1..4, event_json)
  │   ├─ store.save_message(session_id, "user", task)
  │   └─ store.save_message(session_id, "assistant", answer)
  │
  └─ CLI 打印结果
      └─ print("✅ Answer (4 steps):" + answer)
```

```text
step = 4
  │
  ▼
llm.generate(prompt)
  └─ FakeLLM 返回预设响应 4：
     '{"type": "final_answer", "thought": "...", "answer": "## 项目结构分析\\n..."}'
  │
  ▼
parser.parse(raw_output)
  └─ return FinalAnswer(answer="## 项目结构分析\n...")
  │
  ▼
isinstance(parsed, FinalAnswer) → True
  │
  ▼
trace.log_step(step=4, parsed_action="final_answer", observation=answer)
context.add_message("assistant", answer)
  │
  ▼
return AgentResult(success=True, answer="## 项目结构分析...", steps_taken=4)
  │
  ▼
_cmd_run 中：
  ├─ store.save_trace(session_id, step=1..4, event_json)
  ├─ store.save_message(session_id, "user", task)
  ├─ store.save_message(session_id, "assistant", answer)
  └─ print("✅ Answer (4 steps):" + answer)
```

### A.7 完整调用栈

```text
main.py → miniclaw.cli.main()
 └─ _cmd_run(args, config)
     ├─ _create_llm_from_config(config)
     │   └─ FakeLLM([响应1, 响应2, 响应3, 响应4])
     │       响应1: tool_call → list_files({"path": "."})
     │       响应2: tool_call → list_files({"path": "src/miniclaw"})
     │       响应3: tool_call → list_files({"path": "src/miniclaw/agent"})
     │       响应4: final_answer → "## 项目结构分析..."
     │
     ├─ _register_default_tools()
     │   └─ ToolRegistry → [ListFiles, ReadFile, WriteFile, RunShell]
     │
     ├─ SQLiteStore(".miniclaw/miniclaw.db").__enter__()
     │   └─ connect() + init_db()
     │
     ├─ AgentLoop(llm, registry).__init__()
     │   └─ executor + parser + context + recovery
     │
     └─ AgentLoop.run("列出当前目录并总结项目结构")
         │
         │  ── Step 1 ──
         ├─ prompt = build_full_prompt(task, schemas, history=[])
         ├─ raw = llm.generate(prompt)  → list_files({"path": "."})
         ├─ parsed = parser.parse(raw)  → ToolCall
         ├─ obs = executor.execute("list_files", {...})  → Observation(success=True)
         └─ context.add_observation(...)
         │
         │  ── Step 2 ──
         ├─ prompt = build_full_prompt(task, schemas, history=[step1])
         ├─ raw = llm.generate(prompt)  → list_files({"path": "src/miniclaw"})
         ├─ parsed = parser.parse(raw)  → ToolCall
         ├─ obs = executor.execute("list_files", {...})  → Observation(success=True)
         └─ context.add_observation(...)
         │
         │  ── Step 3 ──
         ├─ prompt = build_full_prompt(task, schemas, history=[step1, step2])
         ├─ raw = llm.generate(prompt)  → list_files({"path": "src/miniclaw/agent"})
         ├─ parsed = parser.parse(raw)  → ToolCall
         ├─ obs = executor.execute("list_files", {...})  → Observation(success=True)
         └─ context.add_observation(...)
         │
         │  ── Step 4 ──
         ├─ prompt = build_full_prompt(task, schemas, history=[step1, step2, step3])
         ├─ raw = llm.generate(prompt)  → final_answer("## 项目结构分析...")
         ├─ parsed = parser.parse(raw)  → FinalAnswer
         └─ return AgentResult(success=True, answer="...", steps_taken=4)
         │
         ├─ store.save_trace(...)
         ├─ store.save_message(...)
         └─ print("✅ Answer (4 steps)")
```

---

## Part B：Recovery Demo 路径

以下路径来自 `examples/demo_recovery.py` 和 CLI 中的 Recovery Demo 场景，使用专门的 FakeLLM 预设序列。**这不是 Part A 中 `run` 命令的实际路径。**

### B.1 FakeLLM 预设序列（Recovery Demo 专用）

在 Recovery Demo 中，FakeLLM 预设了 5 个响应，其中第 1 个故意使用错误的工具名：

```text
响应 1: tool_call → open_file({"path": "README.md"})     ← 故意错误
响应 2: tool_call → read_file({"path": "README.md"})     ← 修正
响应 3: tool_call → list_files({"path": "src/miniclaw"})
响应 4: tool_call → list_files({"path": "src/miniclaw/agent"})
响应 5: final_answer → "## 项目结构分析..."
```

### B.2 Step 1：调用不存在的工具

```text
step = 1
  │
  ▼
llm.generate(prompt)
  └─ 返回：'{"type": "tool_call", "tool_name": "open_file", "arguments": {"path": "README.md"}}'
  │
  ▼
parser.parse(raw) → ToolCall(tool_name="open_file")
  │
  ▼
registry.get("open_file") → None   ← 工具不存在！
  │
  ▼
进入错误处理分支（agent/loop.py:152）：
  │
  ├─ recovery.handle_unknown_tool("open_file", registry)
  │   └─ 返回："Tool 'open_file' does not exist. Available tools: list_files, read_file, write_file, run_shell."
  │
  ├─ trace.log_step(step=1, tool_name="open_file", error="...")
  ├─ context.add_message("user", "[Recovery] Tool 'open_file' does not exist...")
  └─ error_count = 1
```

### B.3 Step 2：RecoveryManager 介入

```text
step = 2
  │
  ▼
prompt = build_full_prompt(task, schemas, history=[recovery_hint])
  │   history 中包含 Step 1 的恢复提示，LLM 能看到错误信息
  │
  ▼
llm.generate(prompt)
  └─ 返回：'{"type": "tool_call", "tool_name": "read_file", "arguments": {"path": "README.md"}}'
  │       LLM 收到恢复提示后，自动修正了工具名
  │
  ▼
parser.parse(raw) → ToolCall(tool_name="read_file")
  │
  ▼
registry.get("read_file") → ReadFile 实例（存在）
  │
  ▼
executor.execute("read_file", {"path": "README.md"})
  └─ ReadFile.run(path="README.md") → {"content": "# MiniClaw\n...", "truncated": false}
  │
  ▼
Observation(success=True, output={...})
error_count = 0（成功，重置计数）
```

### B.4 Step 3-5：恢复正常执行

Step 3-5 的流程与 Part A 的正常执行相同，不再重复。最终 5 步完成：

```text
Step 1: 🔧 open_file({'path': 'README.md'})        ← 错误
         ❌ Tool 'open_file' does not exist.
Step 2: 🔧 read_file({'path': 'README.md'})         ← 修复
         → {'content': '# MiniClaw\n...'}
Step 3: 🔧 list_files({'path': 'src/miniclaw'})
         → {'entries': [...]}
Step 4: 🔧 list_files({'path': 'src/miniclaw/agent'})
         → {'entries': [...]}
Step 5: ✅ final_answer
```

---

## Part C：模块职责与设计决策

### C.1 模块依赖关系图

```text
main.py
  │
  ├─ imports ─────────────────────────────────────────────┐
  │                                                       │
  ▼                                                       ▼
AgentLoop (agent/loop.py)                      SQLiteStore (storage/sqlite_store.py)
  │                                              │
  ├─ uses ──┐                                    ├─ sessions
  │         │                                    ├─ messages
  │         ▼                                    ├─ memories
  │   ContextManager (agent/context.py)          └─ traces
  │
  ├─ uses ──┐
  │         ▼
  │   RecoveryManager (agent/recovery.py)
  │
  ├─ uses ──┐
  │         ▼
  │   OutputParser (agent/parser.py)
  │
  ├─ uses ──┐
  │         ▼
  │   ToolExecutor (agent/executor.py)
  │     └─ ToolRegistry (tools/registry.py)
  │         ├─ ListFiles
  │         ├─ ReadFile
  │         ├─ WriteFile
  │         └─ RunShell
  │
  ├─ uses ──┐
  │         ▼
  │   TraceLogger (agent/trace.py)
  │
  └─ uses ──┐
            ▼
      BaseLLM (llm/base.py)
        ├─ FakeLLM
        └─ OpenAIClient
```

### C.2 错误是观察值，不是异常

`ToolExecutor.execute()` 永远不抛异常。所有错误封装为 `Observation(success=False, error=...)`，回到上下文让 LLM 自行修正。

### C.3 Parser 严格，Recovery 兜底

Parser 不做模糊匹配。LLM 要么遵守 JSON 协议，要么收到 `RecoveryManager` 的明确修复提示。解析失败时，`RecoveryManager.handle_invalid_json()` 会尝试从乱文本中提取第一个 `{...}` 块。

### C.4 ContextManager 压缩时机

每轮循环开始时检查 `should_compress()`。只在 token 估算超过 `max_context_tokens` 时触发压缩，保留 system 消息 + 最近 `recent_keep` 条消息，中间部分用 summarizer 压缩为一条 summary。

### C.5 error_count 重置逻辑

| 事件 | error_count 变化 |
| --- | --- |
| ParseError | +1 |
| LLM 异常 | +1 |
| 工具不存在 | +1 |
| 工具执行失败 | +1 |
| 工具执行成功 | = 0（重置） |
| 解析成功 | 不变 |

只有工具成功执行才重置。连续错误累积到 `max_errors`（默认 3）后中止。

### C.6 SQLite 四张表关系

```text
sessions (1)
  ├─ has many → messages (N)    ← 完整对话历史
  └─ has many → traces (N)      ← 每步执行日志

memories (独立)                  ← 跨会话的长期记忆
```
