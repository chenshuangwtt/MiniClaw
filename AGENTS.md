# MiniClaw 项目说明

## 项目定位

MiniClaw 是一个用 Python 自建的轻量级 Agent Harness，用于演示如何从零实现 LLM Agent 运行时；不依赖 LangChain、AutoGen 或 CrewAI。

## 架构

```
用户 → CLI → Agent Loop → LLM（fake/openai）
                    ↓
              JSON Parser → Tool Registry → 执行
                    ↑
              Context Manager / Recovery / Memory / Trace
```

## 技术栈

- **Python 3.11+**，由 **uv** 管理
- **依赖**：`openai`、`tiktoken`、`pydantic`、`PyYAML`；`mem0ai` 为可选依赖（`uv sync --extra mem0`）
- **开发工具**：`pytest`、`ruff`
- **不使用重型框架**：明确排除 LangChain、AutoGen、CrewAI

## 项目结构

```
src/miniclaw/           # 主包
  cli.py                # 标准 CLI 入口
  agent/                # Agent 循环、解析、执行、上下文、恢复、追踪
  tools/                # 工具基类、注册表、文件工具、Shell 工具
  storage/              # SQLite 持久化
  memory/               # 记忆后端抽象与 Mem0 适配器
  llm/                  # BaseLLM、FakeLLM、OpenAI-compatible 客户端
tests/                  # pytest 测试集
examples/               # 演示脚本
```

## 常用命令

```bash
# 安装开发依赖
uv pip install -e ".[dev]"

# 运行测试
uv run pytest

# 使用 FakeLLM 运行（不需要 API Key）
uv run miniclaw run --llm fake

# 使用 OpenAI-compatible Provider 运行
uv run miniclaw run --llm openai --model gpt-4o

# 恢复前查看完整 Run ID
uv run miniclaw runs list

# 静态检查
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## 约定

- 源码统一放在 `src/miniclaw/`，测试统一放在 `tests/`
- 全部代码使用类型标注（Python 3.11+ 语法）
- Docstring 使用 Google 风格
- 保持简单：不引入元类、描述器或隐式魔法
- 每个模块应可独立测试，离线测试优先使用 FakeLLM

## 开发阶段

当前计划见 `ROADMAP.md`。

## v1.0 开发规则

- `AgentEngine` 是唯一执行核心；`AgentLoop` 与 `AsyncAgentLoop` 仅作为轻量外观。
- 显式选择工具协议：离线测试使用 Portable JSON，OpenAI-compatible Provider 使用 Native 模式。
- 每个工具都必须声明 `ToolCapability`；高风险能力在调用方显式开启前保持拒绝。
- 参数校验、异常、错误载荷、超时和非零 Shell 退出码必须转换为结构化失败。
- Checkpoint 是诊断/恢复记录，不能自动重放文件写入、Shell、网络或其他外部副作用。
- 认为工作完成前必须执行 `uv run pytest`、`uv run ruff format --check src tests` 与 `uv run ruff check src tests`。
