# Agent Development Guide

Use `uv sync --extra dev`, then run code with `uv run`.

```powershell
uv run pytest
uv run ruff format --check src tests
uv run ruff check src tests
uv run miniclaw run --llm fake "Inspect this workspace"
uv run python examples/demo_repository_maintenance.py
```

Choose a protocol deliberately:

```powershell
uv run miniclaw run --llm fake --mode portable "List files"
uv run miniclaw run --llm openai --mode native "Inspect a file"
uv run miniclaw resume RUN_ID --approve-pending --allow-file-write
```

Create tools by subclassing `Tool`, supplying `name`, `description`, a JSON Schema `schema`, a `capability`, and `run`. Keep schemas explicit: required fields, primitive types, enums, and `additionalProperties: false` are validated before a call.

Tests should use `FakeLLM` for deterministic offline behavior. Use `LLMResponse` to test native multiple tool calls and their IDs.
