# MiniClaw 开发路线图

## v1.0.0：本地 Agent 运行时

状态：面向本地学习与演示的版本已完成。

- 统一的 async-first Engine，以及同步/异步外观
- Portable JSON 与 OpenAI-compatible Native Tool Calling
- 多工具调用执行与工具调用 ID 关联
- Schema 校验、超时/错误 Observation、能力策略和审计事件
- 受工作区边界限制的文件工具；文件写入、Shell、网络和外部副作用默认拒绝
- SQLite 会话、Trace、Run、Checkpoint 与范围化长期记忆持久化
- 离线仓库维护演示与 GitHub Actions CI
- 本地 `.env` 支持、完整 Run 查询/恢复体验、持久化多轮 Chat 与 DeepSeek Native Tool 验收

## 明确限制

- 这不是安全沙箱。`SandboxExecutor` 只能收窄环境、路径和命令模式，无法提供 OS/容器级隔离。
- Checkpoint 可恢复取消后的上下文和显式审批；标记为 `in_flight` 的 Checkpoint 永不自动重试，必须人工核对。
- Native Provider pending 工具调用和任意进程内线程取消都需要人工核对或进程隔离。
- Trace 回放是只读展示，不是任务恢复。
- Provider 流式能力位于 `BaseLLM.stream()` / `OpenAIClient.stream()`；CLI 不会将已完成回答伪装为实时流。
- Provider 兼容性依赖账户和模型。DeepSeek 真实验收只是小规模只读 smoke test，不代表通用生产可靠性。

## 暂不计划

- 接入 Agent 框架、Web UI、聊天渠道、分布式执行或隐藏的自动后台任务
- 自动破坏性操作，或绕过权限策略
