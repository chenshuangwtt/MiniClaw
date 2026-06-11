# MiniClaw Roadmap

This roadmap keeps MiniClaw focused as a learning-oriented, interview-ready Agent Runtime.

## Current Version: v0.4 Runtime Hardening

Status: complete and ready for GitHub publishing.

- Package CLI entry point: `miniclaw` via `src/miniclaw/cli.py`
- Compatibility wrapper: `main.py`
- Environment diagnostics: `miniclaw doctor`
- Trace aggregation: `miniclaw trace summary`
- Trace HTML export: `miniclaw trace html`
- Workspace-bound file tools
- Shell tool safety controls
- Tool timeouts and cooperative cancellation
- Optional sandboxed shell execution with path boundaries
- SQLite runtime data stored under `.miniclaw/`
- README / README_CN / code walkthrough updated for GitHub presentation
- Permission policy object for file writes and shell commands
- In-memory tool audit logging
- `MINICLAW_*` environment variable overrides
- Trace replay: `miniclaw trace replay`
- Add examples that demonstrate recovery, memory, and context compression end to end
- Optional vector-backed memory with dependency-free hashed embeddings
- Composite memory backend, conflict handling, and importance decay
- LLM-based memory extractor
- Web/search tool behind explicit permission
- OpenAI-compatible streaming client with a formal `BaseLLM.stream()` interface
- Minimal OpenAPI adapter for JSON/YAML specs
- Minimal stdio MCP adapter
- Add CI workflow for lint and tests
- Publish a small architecture diagram in the README

## Before Publishing

- Run `uv sync --extra dev`
- Run `uv run ruff format --check src/ tests/`
- Run `uv run ruff check src/ tests/`
- Run `uv run pytest`
- Run `uv run miniclaw --version`
- Run `uv run miniclaw doctor`
- Confirm `git status --short --ignored` has no runtime data outside ignored paths
- Do not commit `.venv/`, `.uv-cache/`, `.test-tmp/`, `.pytest_tmp/`, `.idea/`, `.codegraph/`, `.miniclaw/`, `*.db`, or `.env`

## v0.5 Candidate

- Optional persistent vector memory store
- Streaming display mode in the CLI
- More integration examples for OpenAI-compatible local endpoints
- Coverage gate in CI
- Schema migration support if persistent `.miniclaw/*.db` compatibility becomes a goal

## Not Planned

- No LangChain, AutoGen, CrewAI, or other agent frameworks
- No heavy UI layer
- No distributed runtime
- No hidden magic around tool calls or parser recovery
