# MiniClaw

> 一个本地优先、用于学习的轻量级 Python Agent Harness。项目从零实现 Agent 运行时，不依赖 LangChain、AutoGen 或 CrewAI。

MiniClaw 关注的不是“调用一次模型”，而是模型周围真正决定 Agent 是否可控的运行时能力：工具调用、权限、上下文、记忆、错误恢复、执行预算、追踪和可恢复状态。

## 核心能力

- `AgentEngine` 是唯一执行核心；`AgentLoop` 与 `AsyncAgentLoop` 分别提供同步和异步外观。
- 支持 Portable JSON 协议和 OpenAI-compatible Native Tool Calling；Native 模式保留 `tool_call_id` 关联。
- 工具调用经过 Schema 校验、能力权限、超时、异常、业务错误与非零退出码检查，统一产出结构化 Observation。
- `ContextManager` 限制工具结果大小，并在 Native/Portable 两种模式下压缩长上下文。
- SQLite 保存会话、消息、Trace、Run、Checkpoint 与按用户和项目隔离的长期记忆。
- 高风险能力默认拒绝；Checkpoint 不会盲目重放可能已执行的副作用操作。

## 架构

```text
用户 / CLI -> AgentEngine -> LLM
                    |          |
                    |          +-> Portable JSON 或 Native Tool Calls
                    v
          ToolExecutor -> PermissionPolicy -> ToolRegistry
                    |
                    +-> Context / Memory / Trace / SQLite Checkpoint
```

`Context` 是当前会话的短期工作历史；`Memory` 是选择性、跨会话的长期知识；`Checkpoint` 是恢复诊断记录。三者用途不同，不能互相替代。

## 快速开始

```powershell
# 安装开发依赖
uv sync --extra dev

# 离线 FakeLLM，不需要 API Key
uv run miniclaw run "列出当前目录的文件"

# 完整离线 Repository Maintenance 演示
uv run python examples/demo_repository_maintenance.py

# 测试与静态检查
uv run pytest
uv run ruff format --check src tests examples
uv run ruff check src tests examples
```

## DeepSeek 配置与运行

复制 `.env.example` 为本地 `.env` 后填写密钥。`.env` 不会覆盖已存在的进程环境变量，最终优先级为：CLI 参数、进程环境变量、`.env`、`miniclaw.toml`、默认值。

```powershell
# 检查本地配置，不输出密钥
uv run miniclaw doctor

# OpenAI-compatible Native Tool Calling
uv run miniclaw run --llm openai --model deepseek-flash --mode native --max-steps 4 "列出文件并读取 README.md，然后总结。"
```

本机验收中，`api.deepseek.com` 上的 `deepseek-flash` 已完成基础请求、只读目录/文件多工具调用，以及未知工具错误后的恢复。模型可用性和名称依赖账户权限，不应把一次 smoke test 视为生产可靠性承诺。

## 常用命令

```powershell
# 交互式多轮会话；同一会话保留 Context、Trace 和可选 Memory
uv run miniclaw chat --llm openai --mode native --memory --user-id alice

# 管理本地长期记忆
uv run miniclaw memory list --user-id alice
uv run miniclaw memory search "偏好" --user-id alice
uv run miniclaw memory delete "精确的记忆文本" --user-id alice

# 查看完整 Run ID，再恢复可恢复任务
uv run miniclaw runs list
uv run miniclaw resume FULL_RUN_ID

# 只读地复盘历史 Trace
uv run miniclaw trace summary
uv run miniclaw trace replay
```

命令退出码：`0` 表示完成；`1` 表示 Agent 失败；`2` 表示参数或配置错误；`3` 表示任务暂停并等待人工审批。CI 或 PowerShell 脚本应检查 `$LASTEXITCODE`。

Windows CI 为 CLI 子进程显式使用 UTF-8；离线集成测试固定使用 FakeLLM、临时工作目录和独立 SQLite 数据库，不会读取本地 DeepSeek 配置或发起网络请求。相同 Chat Session 的 Trace 按 SQLite 插入顺序复盘，因此多个 Run 重复从 Step 1 计数时仍保持真实执行顺序。

## 安全与恢复边界

- 文件写入、Shell、网络和外部副作用默认拒绝，必须在单次运行中显式授权。
- `SandboxExecutor` 只是进程级最佳努力限制，不是操作系统安全沙箱。
- `in_flight` Checkpoint 与 Native pending 调用必须人工核对外部状态，不能自动重试。
- Trace、Audit 和 Checkpoint 会递归脱敏常见 key、token、密码和 Authorization 值，也会处理 Native Tool Calling 的 JSON 字符串参数；这不是完整的 DLP 系统。含敏感参数的待恢复操作不会使用脱敏后的值自动重放，必须重新授权或人工处理。
- `mem0ai` 是可选依赖：`uv sync --extra mem0`。默认长期记忆使用 SQLite。

## 阅读顺序

1. [开发指南](docs/AGENT_DEVELOPMENT_GUIDE.md)
2. [架构说明](docs/ARCHITECTURE.md)
3. [源码执行流程](CODE_WALKTHROUGH.md)
4. [Demo 指南](docs/DEMO_GUIDE.md)
5. [安全与可靠性边界](docs/SECURITY_AND_RELIABILITY.md)

## 许可证

MIT，详见 [LICENSE](LICENSE)。
