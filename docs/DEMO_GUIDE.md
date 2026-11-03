# 演示指南

## 离线仓库维护演示

这是最推荐的首次运行。它使用 `FakeLLM`，将故意损坏的 fixture 复制到临时工作区，读取文件、在显式授权后写入修复、运行 fixture 测试并输出 Trace。

```powershell
uv run python examples/demo_repository_maintenance.py
```

## DeepSeek 原生工具调用

将 `.env.example` 复制为 `.env`，只在本地填写密钥，并保持 `.env` 被忽略。DeepSeek 使用 OpenAI-compatible 接口，可配置 `OPENAI_API_KEY`、`OPENAI_BASE_URL=https://api.deepseek.com`、`MINICLAW_LLM_PROVIDER=openai`，模型名称以当前账户实际返回结果为准。

```powershell
uv run miniclaw doctor
uv run miniclaw run --llm openai --model deepseek-flash --mode native --max-steps 4 "列出文件并读取 README.md，然后总结。"
```

检查类演示应保持只读：不要添加 `--allow-file-write`、`--allow-shell` 或 `--allow-network`。

## 查询与恢复

```powershell
uv run miniclaw runs list
uv run miniclaw trace replay
uv run miniclaw resume FULL_RUN_ID
```

只有 Portable 模式中等待审批的任务可配合 `--approve-pending` 和对应能力开关恢复。标记为 `in_flight` 的任务必须先人工确认外部副作用，不能直接重试。
