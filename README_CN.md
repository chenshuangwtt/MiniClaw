# MiniClaw

> **实验性 / 学习向 / 本地优先** —— 从零实现的 Agent Harness，用于理解 LLM 运行时的工作原理。非生产就绪。不依赖 LangChain、AutoGen、CrewAI。

MiniClaw 实现了 LLM 外围的完整运行时层：Agent 主循环、工具调用协议、上下文压缩、错误恢复、持久化记忆。模型是插件，运行时才是核心。

## 为什么做这个项目

调用 LLM API 很简单，构建一个**能可靠完成任务的 Agent** 才是难点。

难点不在模型本身，而在模型之外的运行时：

- **如何编排多步推理？** → Agent Loop
- **如何让模型调用外部能力？** → Tool Calling Protocol
- **如何处理模型的错误输出？** → Recovery Manager
- **如何在有限的上下文窗口内工作？** → Context Compression
- **如何跨会话持久化知识？** → SQLite Memory

MiniClaw 将以上每个问题实现为独立、可测试的模块。整个系统运行在单机上，除 LLM 端点外不依赖任何外部服务，通过 490+ 项单元测试。

## 入口说明

- `miniclaw`：安装后的控制台命令，真实实现位于 `src/miniclaw/cli.py`。
- `uv run python main.py ...`：源码目录下的兼容入口，方便本地 demo。

标准 CLI 入口是 `src/miniclaw/cli.py`。根目录 `main.py` 只做转发，因此两种命令会走同一套运行时。

## v0.4 可选扩展

- `miniclaw doctor` 检查本地 Python、配置、API key、可选依赖和数据库目录。
- `miniclaw trace summary` 汇总 trace 的步数、结果、错误数量和工具使用次数。
- `miniclaw trace replay` 按步骤复盘历史 session，方便调试。
- 内置文件工具支持 workspace 边界。CLI 默认将 `Path.cwd()` 作为允许访问的根目录。
- `PermissionPolicy` 控制文件写入和 Shell 命令，`AuditLogger` 记录工具调用决策。
- `MINICLAW_*` 环境变量可覆盖配置，不需要直接修改 `miniclaw.toml`。
- `web_search` 已接入显式 `allow_search` 权限，默认关闭。
- `BaseLLM.stream()` 将流式输出变成正式接口，`OpenAIClient` 支持 OpenAI 兼容流式返回。
- `VectorMemoryBackend` 提供无外部依赖的可选向量检索记忆，适合 demo 和测试。
- GitHub Actions 会运行 lint、格式检查和多 Python 版本测试。
- `miniclaw.agent_loop` 等顶层旧模块保留用于早期 demo 兼容；新代码优先使用 `miniclaw.agent`、`miniclaw.tools`、`miniclaw.storage`、`miniclaw.memory`。

## 架构

```text
┌─────────────────────────────────────────────────────────────┐
│                        用户任务                              │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                      Agent Loop                              │
│  ┌─────────┐   ┌─────────┐   ┌────────────┐   ┌──────────┐ │
│  │ Prompt  │──▶│   LLM   │──▶│   Parser   │──▶│ Executor │ │
│  │ Builder │   │         │   │            │   │          │ │
│  └─────────┘   └─────────┘   └────────────┘   └──────────┘ │
│       ▲                                       │             │
│       │            ┌────────────┐             │             │
│       └────────────│  Context   │◀────────────┘             │
│                    │  Manager   │                           │
│                    └────────────┘                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │           Tool Registry              │
        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌─────┐│
        │  │ list │ │ read │ │write │ │shell││
        │  │_files│ │_file │ │_file │ │     ││
        │  └──────┘ └──────┘ └──────┘ └─────┘│
        └──────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │          SQLite Storage              │
        │  sessions · messages · memories      │
        │  · traces                            │
        └──────────────────────────────────────┘
```

## Agent Loop

核心是一个**带有限状态解析器的 while 循环**：

```text
while step < max_steps:
    prompt   = build(system, tools, history, task)
    raw      = llm.generate(prompt)
    output   = parser.parse(raw)          # → ToolCall | FinalAnswer | ParseError

    match output:
        FinalAnswer  → 返回最终答案
        ToolCall     → executor.run(tool, args) → Observation
                     → 追加到 history → 继续循环
        ParseError   → recovery.handle() → 注入修复提示 → 继续循环

    if error_count >= max_errors:
        return 中止
```

关键设计决策：

- **解析器是严格的。** LLM 必须返回恰好一个带 `type` 字段的 JSON 对象。没有正则，没有模糊匹配——模型要么遵守协议，要么收到恢复提示。
- **错误是观察值，不是异常。** 工具失败、解析错误、未知工具全部作为上下文反馈，让模型自我修正。
- **循环永不崩溃。** 每种失败模式都被捕获并返回结构化结果。

## 工具调用协议

LLM 通过 JSON 与工具系统通信：

```json
{
    "type": "tool_call",
    "thought": "我需要先检查目录内容。",
    "tool_name": "list_files",
    "arguments": {"path": "."}
}
```

最终答案使用相同的信封格式：

```json
{
    "type": "final_answer",
    "thought": "我已经看到了文件列表，现在可以总结了。",
    "answer": "项目包含 12 个 Python 文件，组织为..."
}
```

工具以 Python 类的形式注册，附带 JSON Schema：

```python
class ListFiles(Tool):
    name = "list_files"
    description = "列出指定路径下的文件和目录。"
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, path: str, **kwargs) -> dict:
        ...
```

## 恢复机制

Agent 永远不会因为错误输出而崩溃。每种失败类型都有针对性的恢复策略：

| 失败类型 | 恢复策略 |
| --- | --- |
| **非法 JSON** | 通过括号计数提取第一个 `{...}` 块；失败则注入格式提示 |
| **未知工具** | 返回可用工具列表，提醒模型使用其中之一 |
| **参数缺失/类型错误** | 返回工具的 JSON Schema + 验证错误信息 |
| **工具执行失败** | 将错误作为 observation 返回，建议重试或给出 final_answer |
| **连续失败** | 超过 N 次错误后，生成 final_answer 说明失败原因 |

每条恢复消息都写得**机器可读**——它会回到上下文中，让 LLM 在下一轮自行修正。

## 上下文压缩

当估算的 token 数超过 `max_context_tokens` 时：

1. **System 消息**始终保留。
2. **最近 N 条消息**（`recent_keep`）被钉住——它们代表当前的推理状态。
3. 中间的所有消息被**压缩**为一条 `summary` 消息。

默认的摘要器基于规则，不需要调用 LLM：

```text
[Compressed 15 messages]
Roles: {'user': 5, 'assistant': 5, 'tool': 5}
Tools used: get_weather, calculator
Last user: What's the weather in Beijing?
Last assistant: Called get_weather({"city": "Beijing"})
```

替换为 LLM 摘要器只需一个函数：

```python
def llm_summarizer(messages):
    return llm.generate(f"总结这段对话：\n{messages}")

ctx = ContextManager(summarizer=llm_summarizer)
```

## 持久化记忆

所有状态通过 `SQLiteStore` 存储在 SQLite 中：

| 表 | 用途 |
| --- | --- |
| `sessions` | 每次对话一个会话 |
| `messages` | 每个会话的完整消息历史 |
| `memories` | 带重要度排序的键值对长期记忆 |
| `traces` | 逐步事件日志，用于调试 |

```python
with SQLiteStore(".miniclaw/miniclaw.db") as store:
    sid = store.create_session("天气任务")
    store.save_message(sid, "user", "北京天气怎么样？")
    store.save_memory("user:city", "Beijing", importance=5)
    results = store.search_memories("beijing")
```

## 长期记忆：Mem0 集成

LLM API 本身是无状态的——每次请求都从零开始。Agent Harness 需要一个外部记忆层来跨会话携带知识。

MiniClaw 使用**两套存储系统**，各司其职：

| 层 | 存储内容 | 引擎 |
| --- | --- | --- |
| **SQLite** | sessions、messages、traces — 结构化执行日志 | `sqlite3`（标准库） |
| **Mem0** | 用户偏好、项目事实、长期自然语言记忆 | `mem0ai`（语义检索） |

> **SQLite 记录发生了什么；Mem0 记住什么将来有用。**

### 使用方式

```bash
# 保存记忆
uv run miniclaw memory add "用户偏好中文解释 Agent 底层流程" --user-id michael

# 搜索记忆
uv run miniclaw memory search "用户喜欢什么回答风格？" --user-id michael

# 启用记忆运行
uv run miniclaw run "讲一下 RecoveryManager" --user-id michael --memory

# 禁用记忆运行
uv run miniclaw run "task" --no-memory
```

### AgentLoop 内部流程

```text
User Task
    │
    ▼
Memory Search ─── mem0.search(task, user_id) → 相关记忆
    │
    ▼
Memory Injection ─── "## Long-Term Memory\n- 用户偏好中文解释..."
    │
    ▼
LLM Decision ─── 模型在上下文中看到记忆，按需使用
    │
    ▼
Tool Execution → Observation
    │
    ▼
Final Answer
    │
    ▼
Memory Extraction ─── MemoryExtractor.should_remember(task)?
    │
    ▼
Memory Add ─── mem0.add(extracted_text, user_id)
```

记忆搜索失败不会导致 Agent 崩溃。记忆保存失败会被静默记录。

## 快速开始

```bash
# 安装依赖
uv pip install -e ".[dev]"

# 使用 FakeLLM（不需要 API key）
uv run miniclaw run "列出当前目录的文件"

# 使用 OpenAI
export OPENAI_API_KEY=sk-...
uv run miniclaw run --llm openai "总结项目结构"

# 交互模式
uv run miniclaw chat --llm openai

# 查看记忆和追踪
uv run miniclaw memory list
uv run miniclaw trace list

# 导出 trace 为结构化 JSON
uv run miniclaw trace export --session 1 -o trace.json

# 演示：上下文压缩
uv run miniclaw demo context-compression

# 环境健康检查
uv run miniclaw doctor

# 查看最近会话的 trace 汇总
uv run miniclaw trace summary

# 复盘最近会话的 trace 事件
uv run miniclaw trace replay
```

## 配置

`miniclaw.toml` 是项目本地配置。CLI 参数优先级最高，`MINICLAW_*` 环境变量可覆盖默认值和配置文件。

常用环境变量：

```bash
MINICLAW_LLM_PROVIDER=openai
MINICLAW_MODEL=gpt-4o-mini
MINICLAW_API_KEY=sk-...
MINICLAW_DB_PATH=.miniclaw/dev.db
MINICLAW_ALLOW_FILE_WRITE=false
MINICLAW_ALLOW_SHELL=false
MINICLAW_SHELL_ALLOWED_PREFIXES=echo,python -m pytest
```

## 运行演示

### 成功执行 + 错误恢复

```text
$ uv run miniclaw run -v "列出当前目录并总结项目结构"

🐾 MiniClaw v0.4.0 — session #1
📋 Task: 列出当前目录并总结项目结构

✅ Answer (5 steps):

## 项目结构分析
**MiniClaw** 是一个从零实现的轻量级 Agent Harness...
- src/miniclaw/agent/ — Agent 核心
- src/miniclaw/tools/ — 工具系统
...

── Trace ──
  Step 1: 🔧 open_file({'path': 'README.md'})
         ❌ Tool 'open_file' does not exist. Available tools: list_files, read_file, run_shell, write_file.
  Step 2: 🔧 read_file({'path': 'README.md'})
         → {'path': '...', 'content': '# MiniClaw\n\nA lightweight...'}
  Step 3: 🔧 list_files({'path': 'src/miniclaw'})
         → {'entries': [...]}
  Step 4: 🔧 list_files({'path': 'src/miniclaw/agent'})
         → {'entries': [...]}
  Step 5: ✅ final_answer
```

Agent 在 Step 1 调用了不存在的 `open_file`，RecoveryManager 返回可用工具列表后，Agent 在 Step 2 自动修正为 `read_file`。错误记录在 Trace 中，但不会导致循环崩溃。

## 测试

```bash
# 运行全部测试
uv run pytest

# 带覆盖率
uv run pytest --cov=miniclaw --cov-report=term-missing

# 运行指定模块
uv run pytest tests/test_agent_loop.py -v
```

```text
============================ 466 passed in ... ============================
```

pytest 已配置使用 `.test-tmp` 作为临时目录。如果 Windows 留下异常 ACL，删除该目录后重新运行 `uv run pytest`。

## 项目结构

```text
MiniClaw/
├── main.py                          # 本地运行兼容包装
├── pyproject.toml                   # 项目元数据和依赖
├── requirements.txt
├── miniclaw.toml                    # 本地配置示例（已 gitignore）
├── ROADMAP.md                       # 开发路线图
├── CODE_WALKTHROUGH.md             # 执行流程解读
│
├── src/miniclaw/
│   ├── cli.py                       # 标准 CLI 入口
│   │
│   ├── agent/                       # 核心 Agent 运行时
│   │   ├── loop.py                  # AgentLoop — 核心 while 循环
│   │   ├── state.py                 # Pydantic 模型：ToolCall, FinalAnswer
│   │   ├── parser.py                # JSON 解析器 + 校验
│   │   ├── executor.py              # ToolExecutor + Observation
│   │   ├── prompts.py               # Prompt 构建器（含记忆注入）
│   │   ├── context.py               # ContextManager 上下文压缩
│   │   ├── recovery.py              # RecoveryManager（5 种恢复策略）
│   │   ├── trace.py                 # StepTrace + TraceLogger
│   │   └── config.py                # TOML 配置加载器
│   │
│   ├── tools/                       # 工具系统
│   │   ├── base.py                  # Tool 抽象基类
│   │   ├── registry.py              # ToolRegistry
│   │   ├── file_tools.py            # list_files, read_file, write_file
│   │   ├── shell_tool.py            # run_shell（带安全过滤）
│   │   ├── search_tool.py           # web_search（权限门控）
│   │   ├── permissions.py           # PermissionPolicy（默认拒绝）
│   │   ├── audit.py                 # 工具执行审计日志
│   │   └── security.py              # 工作区路径解析
│   │
│   ├── llm/                         # LLM 抽象层
│   │   ├── base.py                  # BaseLLM 抽象接口
│   │   ├── fake.py                  # FakeLLM 测试替身
│   │   ├── openai_client.py         # OpenAI 客户端（支持 streaming）
│   │   └── openai.py                # 旧版 OpenAI 客户端
│   │
│   ├── memory/                      # 记忆抽象层
│   │   ├── base.py                  # MemoryBackend + NullMemoryBackend
│   │   ├── extractor.py             # MemoryExtractor（关键词提取）
│   │   ├── mem0_store.py            # Mem0MemoryBackend（语义搜索）
│   │   └── vector.py                # VectorMemoryBackend（内存向量）
│   │
│   ├── storage/                     # 持久化存储
│   │   ├── sqlite_store.py          # SQLite sessions/messages/memories/traces
│   │   └── memory.py                # 旧版 SQLite 键值存储
│   │
│   └── multiagent/                  # 多 Agent 原型
│       ├── agents.py                # PlannerAgent, CoderAgent, ReviewerAgent
│       └── coordinator.py           # Coordinator（顺序调度）
│
└── tests/                           # 25 个测试文件，490+ 个测试用例
```

## 简历亮点

- **从零设计并实现了完整的 LLM Agent Runtime**——包括 Agent 主循环、工具调用协议、上下文压缩、错误恢复和持久化记忆——未依赖任何现有 Agent 框架（LangChain、AutoGen、CrewAI）。

- **构建了结构化的工具调用协议**：使用 Pydantic 校验 JSON Schema、注册表模式管理工具发现、恢复管理器将 LLM 的畸形输出、未知工具调用和执行失败转化为自纠正反馈循环。

- **实现了上下文管理与持久化记忆**：采用滑动窗口压缩策略配合规则摘要，使用 SQLite 存储会话、消息、长期记忆和执行追踪，并加入可选向量检索，通过 490+ 项单元测试覆盖全部模块。

- **基于 Mem0 扩展长期语义记忆层**：在任务开始前根据用户请求检索相关偏好和项目事实并注入上下文，任务结束后抽取高价值记忆持久化；结合 SQLite 结构化存储 sessions/traces，实现短期执行状态与长期用户记忆解耦。

## Future Work

See `ROADMAP.md`.

## License

MIT
