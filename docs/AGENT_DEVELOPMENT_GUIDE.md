# Agent 开发指南

## 从这里开始

LLM 只会根据当前请求生成文本；Agent 则在模型外增加一个循环：选择动作、接收 Observation、再决定下一步。MiniClaw 将这个循环显式写在代码中，便于学习和调试。

```powershell
uv sync --extra dev
uv run pytest
uv run ruff format --check src tests
uv run ruff check src tests
```

## Agent 循环

`AgentEngine.run_async()` 会创建可持久化的 Run、记录用户回合、检索范围化记忆、调用模型、按授权执行工具、保存 Observation，并持续循环，直到产生最终回答或触发预算/错误限制。`max_steps`、`max_errors`、工具调用次数、重复调用次数、超时和上下文预算共同确保循环有限。

## 两种工具协议

- **Portable JSON**：确定性强，适合 `FakeLLM` 离线测试。模型输出 `{"type":"tool_call", ...}` 或 `{"type":"final_answer", ...}`。
- **Native**：使用 OpenAI-compatible Provider 的 `tool_calls`。MiniClaw 会在后续 `role=tool` 消息中保留对应调用 ID，适用于 DeepSeek 等 Provider。

## 编写工具

继承 `Tool`，声明精确的 JSON Schema 和 `ToolCapability`，再实现 `run`。`ToolExecutor` 会在调用前拒绝缺失或不合法的参数。优先使用 `READ_ONLY`；文件写入、Shell、网络和外部副作用默认拒绝。

```python
class Echo(Tool):
    name = "echo"
    description = "返回输入文本。"
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    capability = ToolCapability.READ_ONLY

    def run(self, text: str) -> str:
        return text
```

## Context、Memory 与恢复

`ContextManager` 服务于单个 Engine/会话：保留近期回合、限制工具结果，并在必要时将旧历史压缩为规则摘要。`MemoryManager` 管理独立的长期记忆：默认 SQLite 后端只保存被抽取器选中的片段，过滤常见敏感信息，并按用户和工作区隔离。可用 `miniclaw memory list/search/add/delete` 查看和管理。

`SQLiteStore` 保存 Run 和 Checkpoint。`miniclaw runs list` 会展示完整 ID。恢复只处理非终态任务；`in_flight` 副作用和 Native pending 调用必须由人先确认外部状态。这是刻意的设计：Checkpoint 不是 exactly-once 事务。

## 调试流程

1. 先用 `FakeLLM` 和聚焦测试稳定复现。
2. 通过 `result.trace` 或 `miniclaw trace replay` 检查“决策 -> Observation”链路。
3. 在调整提示词前，先确认是否是权限或 Schema 失败。
4. 真实 Provider 测试保持小规模、只读，并明确标注为真实 API 结果。
5. 修改代码或文档后，运行完整离线测试集。
