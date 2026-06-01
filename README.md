# 🐾 MiniClaw

A **lightweight, self-built Agent Harness** in Python — no LangChain, no AutoGen, no CrewAI.

MiniClaw demonstrates how to build a fully functional LLM agent runtime from scratch with minimal dependencies.

## Features

| Module | What it does |
|--------|-------------|
| **Agent Loop** | Core engine: LLM ↔ Tool conversation loop |
| **Tool Registry** | Register tools with decorators, auto-generate JSON Schema |
| **JSON Parser** | Robustly extract tool calls from LLM free-form text |
| **Context Manager** | Token-budget trimming with pinned turns |
| **Recovery Manager** | Exponential backoff retry + repair prompts |
| **Memory** | SQLite-backed conversation history & key-value store |
| **Trace Logger** | Structured JSONL traces for debugging |
| **FakeLLM** | Programmable mock LLM for offline development |
| **OpenAI Client** | Works with OpenAI API, Ollama, vLLM, etc. |
| **CLI** | Interactive terminal interface |

## Quick Start

```bash
# Install
pip install -e ".[dev]"

# Run with FakeLLM (no API key needed)
miniclaw run --llm fake

# Run with OpenAI
miniclaw run --llm openai --model gpt-4o

# Run tests
pytest
```

## Architecture

```
User
 │
 ▼
┌─────────┐
│   CLI   │
└────┬────┘
     ▼
┌──────────────┐    ┌───────────────┐
│  Agent Loop  │───▶│ Context Mgr   │
└──────┬───────┘    └───────────────┘
       │
       ├──────────────────┐
       ▼                  ▼
┌─────────────┐    ┌─────────────┐
│    LLM      │    │ Trace Logger │
│ (fake/      │    └─────────────┘
│  openai)    │
└──────┬──────┘
       │ raw response
       ▼
┌──────────────┐
│ JSON Parser  │
└──────┬──────┘
       │
       ├── text → return to user
       │
       ├── tool_calls → Tool Registry → execute → loop back
       │
       ▼ (on error)
┌──────────────┐
│  Recovery    │
└──────────────┘
```

## Why MiniClaw?

Most agent frameworks hide the runtime behind layers of abstraction. MiniClaw strips it down to the essentials:

- **~500 lines of core code** — read it all in one sitting
- **Zero framework dependencies** — just `openai` and `tiktoken`
- **Fully testable offline** — FakeLLM lets you develop without API keys
- **Production patterns** — retry logic, context management, structured tracing

## License

MIT
