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
  agent_loop.py         # Core Agent class — the engine
  tool_registry.py      # Tool registration + execution
  json_parser.py        # Extract JSON/tool-calls from LLM text
  context.py            # Token-budget message trimming
  recovery.py           # Retry + repair-prompt injection
  memory.py             # SQLite key-value + message history
  trace.py              # Structured trace logging
  cli.py                # CLI entry point
  llm/
    base.py             # BaseLLM abstract interface
    fake.py             # FakeLLM for testing
    openai.py           # OpenAI-compatible client
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

1. ✅ **Phase 1 — Foundation**: tool_registry, json_parser, llm/base, llm/fake
2. **Phase 2 — Core Loop**: agent_loop (minimal)
3. **Phase 3 — Hardening**: recovery, context, trace
4. **Phase 4 — Persistence & Real LLM**: memory, llm/openai, cli
5. **Phase 5 — Polish**: examples, docs
