# 更新日志

## 1.0.0

- 修复 Native Tool Calling 嵌套 JSON 参数的递归脱敏；含敏感参数的待恢复调用必须重新授权，不能用脱敏值重放。
- 修复 CLI 退出状态：成功为 `0`、Agent 失败为 `1`、参数/配置错误为 `2`、等待审批为 `3`。
- 最终运行时加固：本地 `.env` 及明确优先级、完整 Run ID 和 `runs list`、持久化多轮 Chat、Native 请求预算压缩、MCP `isError` 处理与诊断记录脱敏。
- `mem0ai` 调整为可选依赖；SQLite 继续作为默认长期记忆后端。
- 使用 `deepseek-flash` 验证了本地 DeepSeek OpenAI-compatible 端点的基础 Native 完成、只读多工具调用和未知工具恢复。
- 加入统一的 async-first Agent Engine 与明确的 Run 生命周期状态。
- 加入 Native 多工具调用执行和 Provider 工具调用 ID 关联。
- 加入基于能力的权限、实用 JSON Schema 校验、结构化失败和非零 Shell 退出处理。
- 加入 SQLite Run 记录与 Checkpoint。
- 加入隔离的离线仓库维护演示，以及 v1 架构、安全、开发、源码导读和 Demo 文档。
- 移除 CLI 对已完成回答的伪流式输出；Provider 流式能力仍通过 LLM API 提供。

## 0.4.0

- 加入运行时加固、Trace 工具、适配器、记忆扩展和 CI。
