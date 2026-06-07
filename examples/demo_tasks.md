# MiniClaw Demo 展示

三个可运行的 Demo，分别展示 MiniClaw 的核心能力。

---

## Demo 1：项目结构分析

### 技术亮点

多工具编排 + 结构化 Agent Loop

### 用户输入

```text
分析当前目录的项目结构，列出所有 Python 文件，并统计代码行数。
```

### Agent 预期执行步骤

```text
Step 1: Agent 决定先列出目录
        → tool_call: list_files({"path": "."})

Step 2: 看到 src/ 目录，决定深入
        → tool_call: list_files({"path": "src/miniclaw"})

Step 3: 选择一个文件读取
        → tool_call: read_file({"path": "src/miniclaw/agent/loop.py"})

Step 4: 统计行数
        → tool_call: run_shell({"command": "find . -name '*.py' | xargs wc -l"})

Step 5: 综合所有观察，给出 final_answer
```

### FakeLLM 模拟脚本

```python
from miniclaw.llm.fake import FakeLLM

llm = FakeLLM([
    # Step 1: 列出根目录
    json.dumps({
        "type": "tool_call",
        "thought": "先看看项目根目录有什么文件。",
        "tool_name": "list_files",
        "arguments": {"path": "."}
    }),
    # Step 2: 深入 src 目录
    json.dumps({
        "type": "tool_call",
        "thought": "根目录有 src/ 和 tests/，先看 src/miniclaw 的结构。",
        "tool_name": "list_files",
        "arguments": {"path": "src/miniclaw"}
    }),
    # Step 3: 读取关键文件
    json.dumps({
        "type": "tool_call",
        "thought": "看到 agent/ 子目录，读取 loop.py 了解核心逻辑。",
        "tool_name": "read_file",
        "arguments": {"path": "src/miniclaw/agent/loop.py"}
    }),
    # Step 4: 统计行数
    json.dumps({
        "type": "tool_call",
        "thought": "现在统计所有 Python 文件的代码行数。",
        "tool_name": "run_shell",
        "arguments": {"command": "find . -name '*.py' | head -20"}
    }),
    # Step 5: 最终回答
    json.dumps({
        "type": "final_answer",
        "thought": "我已经收集了足够的项目结构信息。",
        "answer": "## 项目结构分析\n\n..."
    }),
])
```

### 运行命令

```bash
uv run python examples/demo_project_analysis.py
```

### 展示能力

- **多步推理**：Agent 自主决定探索路径（根目录 → src → agent/ → 统计）
- **工具编排**：4 次工具调用，每次结果影响下一步决策
- **结构化输出**：final_answer 包含 Markdown 格式的分析报告

---

## Demo 2：异常工具调用恢复

### 技术亮点

RecoveryManager 自纠正 + 容错能力

### 用户输入

```text
读取 README.md 文件的内容。
```

### Agent 预期执行步骤

```text
Step 1: Agent 调用了不存在的工具 open_file
        → tool_call: open_file({"path": "README.md"})
        → 工具不存在！RecoveryManager 介入
        → 返回: "Tool 'open_file' does not exist. Available tools: ..."
        → 错误注入到上下文

Step 2: Agent 收到恢复提示，自我修正
        → tool_call: read_file({"path": "README.md"})
        → 成功！返回文件内容

Step 3: Agent 给出 final_answer
```

### FakeLLM 模拟脚本

```python
from miniclaw.llm.fake import FakeLLM

llm = FakeLLM([
    # Step 1: 故意使用错误的工具名
    json.dumps({
        "type": "tool_call",
        "thought": "我需要读取文件内容。",
        "tool_name": "open_file",       # ← 不存在的工具！
        "arguments": {"path": "README.md"}
    }),
    # Step 2: 收到恢复提示后，修正为正确的工具名
    json.dumps({
        "type": "tool_call",
        "thought": "抱歉，工具名是 read_file 不是 open_file。",
        "tool_name": "read_file",       # ← 修正后的正确工具名
        "arguments": {"path": "README.md"}
    }),
    # Step 3: 最终回答
    json.dumps({
        "type": "final_answer",
        "thought": "文件读取成功。",
        "answer": "成功读取 README.md。"
    }),
])
```

### 运行命令

```bash
uv run python examples/demo_recovery.py
```

### 执行过程（带 Trace）

```text
🐾 MiniClaw — Demo 2: Recovery Manager
📋 Task: 读取 README.md 文件的内容。

── Trace ──
  Step 1: 🔧 open_file({'path': 'README.md'})
         ❌ ERROR: Tool 'open_file' does not exist.
  Step 2: 🔧 read_file({'path': 'README.md'})
         → {'path': '...', 'content': '# MiniClaw\n\nA lightweight...', ...}
  Step 3: ✅ final_answer

✅ Answer (3 steps):
成功读取 README.md。
```

### 展示能力

- **零崩溃**：调用不存在的工具不会让 Agent 崩溃，RecoveryManager 返回结构化错误
- **自纠正**：错误信息注入上下文后，Agent 在下一轮自动修正工具名
- **可观测**：Trace 完整记录了错误 → 恢复 → 成功的全过程

---

## Demo 3：上下文压缩

### 技术亮点

ContextManager 自动压缩 + 长对话支持

### 场景设定

模拟一个长对话：Agent 已经执行了 15 轮工具调用，积累了大量上下文。此时用户提出新问题，ContextManager 自动触发压缩。

### 代码演示

```python
from miniclaw.agent.context import ContextManager

# 创建一个 token 预算很小的 ContextManager（便于演示）
ctx = ContextManager(max_context_tokens=50, recent_keep=4)

# 模拟长对话历史：15 轮 user + assistant + tool
for i in range(1, 16):
    ctx.add_message("user", f"第 {i} 个问题：分析 module_{i}。")
    ctx.add_message("assistant", f"好的，分析 module_{i}。")
    ctx.add_observation("read_file", {"path": f"module_{i}.py"}, f"module_{i} 代码...")

# 压缩前: 45 条消息, 约 1350 tokens
# 压缩后: 5 条消息, 约 50 tokens
ctx.compress()
```

### 运行命令

```bash
uv run python examples/demo_context_compression.py
```

### 预期输出

```text
🐾 MiniClaw — Demo 3: 上下文压缩

--- 压缩前 ---
消息数: 60, 估算 tokens: 655

--- 压缩后 ---
消息数: 5, 估算 tokens: 97

压缩后的消息列表:
  [0] summary: [Compressed 56 messages]
      Roles: {'user': 14, 'assistant': 28, 'tool': 14}
      Tools used: read_file
  [1] user: 第 15 个问题...
  [2] assistant: 好的，分析 module_15...
  [3] assistant: Called read_file(...)
  [4] tool: module_15 代码...
```

### 展示能力

- **自动触发**：token 估算超过 `max_context_tokens` 时自动压缩
- **保留关键信息**：最近 4 条消息完整保留，代表当前推理状态
- **可替换摘要器**：默认规则摘要（零开销），可无缝替换为 LLM 摘要器

---

## Demo 4：SQLite 持久化记忆

### 技术亮点

四张表读写 + 搜索 + UPSERT + 跨会话隔离

### 运行命令

```bash
uv run python examples/demo_memory.py
```

### 预期输出

```text
[1] 创建了 2 个会话: #1, #2
[2] 会话 #1 有 4 条消息，会话 #2 有 2 条消息
[3] 保存了 4 条长期记忆
[4] 搜索 'alice': 1 条结果 → user:name=Alice
[5] 所有记忆（按重要度排序）:
    [project:name] = MiniClaw  importance=*********
    [user:name] = Alice  importance=********
[6] 会话 #1 保存了 2 条 trace
[7] 会话 #1 的 trace:
    Step 1: tool_call → get_weather({'city': 'Beijing'})
    Step 2: final_answer
[8] UPSERT: user:name 从 Alice 更新为 Bob
[9] 两个会话共 6 条消息，记忆跨会话持久化
```

### 展示能力

- **sessions**：创建/查询，按 ID 隔离
- **messages**：保存/列表/按会话隔离，支持 user/assistant/tool 角色
- **memories**：保存/搜索（LIKE 关键词）/UPSERT/按重要度排序
- **traces**：保存/查询/JSON 往返解析
- **参数化 SQL**：所有写入用 `?` 占位符，零注入风险

---

## 运行全部 Demo

```bash
# 安装依赖
uv pip install -e ".[dev]"

# Demo 1: 项目结构分析
uv run python examples/demo_project_analysis.py

# Demo 2: 异常恢复
uv run python examples/demo_recovery.py

# Demo 3: 上下文压缩
uv run python examples/demo_context_compression.py

# Demo 4: SQLite 持久化记忆
uv run python examples/demo_memory.py
```

所有 Demo 均使用 FakeLLM，**不需要 OpenAI API Key**，离线可跑。
