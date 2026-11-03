# MiniClaw v1.0 源码执行流程

## 1. 入口

`uv run miniclaw run ...` 会进入 `src/miniclaw/cli.py`。`load_config()` 读取 `miniclaw.toml` 与本地 `.env`，但不会覆盖已有进程环境变量；最后以显式 CLI 参数为准。CLI 创建 LLM Provider、工具注册表、权限策略、审计记录器、SQLite 会话和 `AgentLoop`，并打印 `miniclaw resume` 所需的完整 Run ID。

## 2. 一个 Engine，两套 API

`AgentLoop.run()` 是同步外观，`AsyncAgentLoop.run()` 是异步外观。两者都调用 `src/miniclaw/agent/engine.py` 中的 `AgentEngine`，因此共享 Context、重试限制、Trace、取消、生命周期状态、权限处理和结果语义。

```text
task -> AgentEngine -> LLM -> parse/provider tool_calls -> ToolExecutor
                                                ^                |
                                                |                v
                                          Context/trace <- Observation
```

每个回合中，`AgentEngine.run_async()` 只记录一次用户消息，创建一条 `runs` 记录，检索范围化记忆，然后选择 `_run_portable()` 或 `_run_native()`。最终的 `AgentResult` 会持久化生命周期状态和 Checkpoint。

## 3. 工具协议

Portable 模式要求模型输出如下 JSON 信封：

```json
{"type":"tool_call","tool_name":"read_file","arguments":{"path":"README.md"}}
```

Native 模式消费 Provider 返回的 `tool_calls` 数组。每个调用都会保留 assistant 消息，经 `ToolExecutor` 执行后，再追加带相同 `tool_call_id` 的 `role=tool` 消息；同一响应中的多个调用按顺序执行。每次 Native 请求前，`_prepare_native_request()` 会计算实际组装后的 Provider 消息，截断过大的工具 Observation，并让 `ContextManager` 压缩旧历史。压缩不会拆开被保留的 assistant 工具调用与对应工具结果。

## 4. 信任边界

`ToolExecutor` 会校验 Schema、能力策略、超时、异常、返回错误载荷和命令退出码。MCP 的 `isError: true` 会成为失败 Observation。所有失败都是结构化 `Observation`；工具失败未解决时，即使模型给出最终回答，Run 仍会失败，避免假成功。

工具会声明能力：`read_only`、`file_write`、`shell_execution`、`network_access` 和 `external_side_effect`。除只读外均需要显式开启。

## 5. 可持久化运行数据

`SQLiteStore` 保存会话、消息、记忆、Trace、Run 和 Checkpoint。Chat 在所有回合复用同一 Engine 和会话，早期用户约束可继续用于后续回合。Checkpoint 保存恢复状态和审批上下文，但不会自动重放副作用工具；`in_flight` 与 Native pending 调用需要人工核对。Trace、Audit 和 Checkpoint 在持久化前会脱敏常见 API Key、Authorization、Token、密码和 Secret 模式。

## 6. 离线查看

```powershell
uv run python examples/demo_repository_maintenance.py
```

该演示使用 `FakeLLM` 和损坏 fixture 的临时副本：列出文件、读取源码和测试文件、在显式策略下写入修复、运行 fixture 测试并打印 Trace。
