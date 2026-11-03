# 安全与可靠性

MiniClaw 采用本地优先、保守默认值。`PermissionPolicy` 在调用方显式开启前拒绝文件写入、Shell 执行、网络访问和外部副作用。文件工具会将相对路径解析到配置的工作区根目录下，并拒绝越界路径。

默认长期记忆后端是 SQLite。`MemoryManager` 只保存抽取器选中的片段，应用配置的冲突策略，并在持久化前过滤常见 API Key、密码、Token、银行卡和私钥模式。记录按 `user_id` 与工作区派生的项目 ID 隔离；这是单库内的逻辑隔离，不是加密或多租户授权。

`ToolExecutor` 会在调用前校验参数，并将权限失败、Schema 失败、超时、异常、适配器业务错误（包括 MCP `isError: true`）和非零命令退出码转为失败的 `Observation`。后续模型给出最终回答，不能覆盖未解决工具失败。

`RunShell` 有命令黑名单和可选前缀白名单。`SandboxExecutor` 提供允许工作路径、环境净化和超时等最佳努力进程限制；它**不是**操作系统安全边界。不要在其中执行不可信命令或传入凭据；面对恶意负载应使用真实的容器、虚拟机或系统级沙箱。

`miniclaw runs list` 展示完整 Run ID，`miniclaw resume FULL_RUN_ID` 用于恢复。已完成和失败的 Run 不能恢复；取消的 Run 可以从保存的 Checkpoint 恢复。`in_flight` Checkpoint 不会被重放，因为外部操作可能已成功但尚未保存结果。等待审批的 Checkpoint 只有在带 `--approve-pending` 和相应能力开关（如 `--allow-file-write`）时才会继续。

取消是模型/工具步骤间的协作式取消。超时会把控制权交还 Engine，但任意 Python 线程不能被安全地强制终止；需要硬取消时应使用子进程超时或真实进程隔离。Native Provider pending 调用也始终要求人工核对。上下文压缩会保留 Native assistant 工具调用与对应工具结果的合法关联，并在注入前截断过大的工具结果。

Trace、Audit 和 Checkpoint 会递归脱敏常见 API Key、Authorization、Token、密码和 Secret 模式，包括 Native Tool Calling 中 `function.arguments` 这类 JSON 字符串。脱敏为持久化和展示生成副本，不会改写当前运行时的工具参数。这是纵深防御，不应替代“不要将凭据放进工具参数或模型提示词”的基本原则。

若等待恢复的工具调用包含敏感参数，Checkpoint 会保留可诊断的脱敏结构并标记为需要重新授权；恢复流程绝不会使用 `[REDACTED]` 参数自动调用工具。CLI 退出码为：`0` 成功，`1` Agent 失败，`2` 参数或配置错误，`3` 等待人工审批。

不要提交 `.env`、API Key、SQLite 运行数据、生成的 Trace 或 `.venv`。`.gitignore` 已覆盖这些本地文件；发布前仍应检查 `git status --ignored`。
