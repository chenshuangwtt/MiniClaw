# MiniClaw 开发说明

MiniClaw 是本地、学习导向的 Python Agent Harness，刻意不使用 LangChain、AutoGen 和 CrewAI。

- 使用 Python 3.11+ 与 `uv`；通过 `uv run pytest` 运行测试。
- 运行时代码放在 `src/miniclaw`，测试放在 `tests`。
- `AgentEngine` 是同步与异步公开 API 共用的执行核心。
- Portable 模式接受 JSON-in-text；Native 模式遵守 OpenAI-compatible `tool_calls` 与 `tool_call_id` 关联规则。
- 工具按照能力分类。文件写入、Shell、网络与外部副作用必须显式授权。
- 不要把 `SandboxExecutor` 描述为安全边界；它只提供进程级最佳努力限制。
- SQLite Checkpoint 不是重放副作用工具的授权。
- `.env` 会以低于进程环境和 CLI 参数的优先级加载；绝不打印或持久化其中的凭据。
- 使用 `miniclaw runs list` 获取 `miniclaw resume` 所需的完整 ID。
