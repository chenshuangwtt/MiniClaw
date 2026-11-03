# MiniClaw — Project Context

## What is this?

MiniClaw is a **lightweight, self-built Agent Harness** in Python. It demonstrates how to build an LLM agent runtime from scratch — no LangChain, no AutoGen, no CrewAI.

## Architecture

```
User → CLI → Agent Loop → LLM (fake/openai)
                    ↓
              JSON Parser → Tool Registry → execute
                    ↑
              Context Manager / Recovery / Memory / Trace
```

## Tech Stack

- **Python 3.11+**, managed by **uv**
- **Dependencies**: `openai`, `tiktoken` (minimal)
- **Dev**: `pytest`, `ruff`
- **No heavy frameworks**: LangChain/AutoGen/CrewAI are explicitly excluded

## Project Layout

```
src/miniclaw/           # Main package
  cli.py                # Canonical CLI entry point
  agent/                # Agent loop, parser, executor, context, recovery, trace
  tools/                # Tool base class, registry, file tools, shell tool
  storage/              # SQLite persistence
  memory/               # Memory backend abstraction + Mem0 adapter
  llm/                  # BaseLLM, FakeLLM, OpenAI-compatible client
tests/                  # pytest test suite
examples/               # Demo scripts
```

## Commands

```bash
# Install (dev mode)
uv pip install -e ".[dev]"

# Run tests
uv run pytest

# Run with FakeLLM (no API key needed)
uv run miniclaw run --llm fake

# Run with OpenAI
uv run miniclaw run --llm openai --model gpt-4o

# Lint
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

## Conventions

- All source in `src/miniclaw/`, all tests in `tests/`
- Type hints everywhere (Python 3.11+ syntax)
- Docstrings: Google style
- Keep it simple — no metaclasses, no descriptors, no magic
- Each module is independently testable (FakeLLM enables offline testing)

## Development Phases

See `ROADMAP.md` for the current development plan.

## v1.0 Development Rules

- `AgentEngine` is the only execution core. Keep `AgentLoop` and `AsyncAgentLoop` as thin facades.
- Select the tool protocol explicitly: portable JSON for offline testing and native mode for OpenAI-compatible providers.
- Every tool must declare a `ToolCapability`; high-risk capabilities stay denied unless a caller opts in.
- Tool results must become structured failures for validation errors, exceptions, error payloads, timeouts, and non-zero shell exit codes.
- Checkpoints are diagnostic/recovery records. Do not automatically replay file writes, shell calls, network calls, or external side effects.
- Use `uv run pytest`, `uv run ruff format --check src tests`, and `uv run ruff check src tests` before considering work complete.
