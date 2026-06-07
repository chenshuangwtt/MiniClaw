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

Current focus:

1. ✅ **v0.2 Hardening**: package CLI, doctor command, trace summary, workspace-bound file tools, safer shell tool, runtime data under `.miniclaw/`.
2. **v0.3 Permissions & Replay**: explicit permission policy, tool audit logs, trace replay, better examples.
3. **v0.4 Optional Extensions**: vector memory, web/search tool, streaming OpenAI-compatible client, CI.
