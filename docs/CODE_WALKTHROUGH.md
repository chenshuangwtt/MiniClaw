# 源码导读

从 `src/miniclaw/cli.py` 开始：它构造 LLM、工具注册表、权限策略、SQLite 会话和 `AgentLoop`。

`src/miniclaw/agent/engine.py` 包含真正的循环。`_run_portable` 解析 JSON 信封；`_run_native` 处理 Provider 响应和全部原生工具调用。两者都使用 `_execute`、`ToolExecutor`、Trace、Context 和一致的最终结果规则。

`src/miniclaw/agent/executor.py` 是工具信任边界。它在产出 `Observation` 前检查注册表、Schema、能力策略、超时和执行结果。

`src/miniclaw/storage/sqlite_store.py` 管理持久化数据。Run 与 Checkpoint 独立于会话消息，因此诊断回放始终是只读的。
