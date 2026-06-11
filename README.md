# MiniClaw

> **Experimental / Educational / Local-first** — a from-scratch Agent Harness for learning how LLM runtimes work. Not production-ready. No LangChain, no AutoGen, no CrewAI.

MiniClaw implements the full runtime layer around an LLM: the agent loop, tool execution protocol, context compression, error recovery, and persistent memory. The model is a plugin; the runtime is the product.

## Why This Project

Calling an LLM API is trivial. Building an **agent that reliably solves tasks** is not.

The hard part is not the model — it's everything around it:

- **How do you orchestrate multi-step reasoning?** → Agent Loop
- **How do you let the model invoke external capabilities?** → Tool Calling Protocol
- **How do you handle malformed output?** → Recovery Manager
- **How do you stay within the context window?** → Context Compression
- **How do you persist knowledge across sessions?** → SQLite Memory

MiniClaw implements each of these as a discrete, testable module. The result is a minimal but complete agent runtime that runs on a single machine, needs no external services beyond an LLM endpoint, and passes 660+ unit tests.

## Entry Points

- `miniclaw` — installed console script, implemented by `src/miniclaw/cli.py`.
- `uv run python main.py ...` — source-tree compatibility wrapper for local demos.

The canonical CLI lives in `src/miniclaw/cli.py`. The root `main.py` only forwards to that package entry point so both commands exercise the same runtime.

## v0.4 Runtime Hardening

- `miniclaw doctor` checks the local Python, config, API key, optional packages, and database directory.
- `miniclaw trace summary` aggregates stored traces by step count, result, errors, and tool usage.
- `miniclaw trace replay` replays a stored session step by step for debugging.
- `miniclaw trace html` exports a self-contained HTML report with a Mermaid trace timeline.
- Built-in file tools can enforce a workspace boundary. The CLI registers file tools with `Path.cwd()` as the allowed root.
- `PermissionPolicy` gates file writes and shell commands, and `AuditLogger` records tool execution decisions.
- `MINICLAW_*` environment variables can override config values without editing `miniclaw.toml`.
- `web_search` is available behind explicit `allow_search` permission.
- `BaseLLM.stream()` formalizes streaming output; `OpenAIClient` implements OpenAI-compatible token streaming.
- `ToolExecutor` supports wall-clock tool timeouts and cooperative cancellation.
- `SandboxExecutor` adds restricted shell execution with command blocklists, environment sanitization, and path boundaries.
- `VectorMemoryBackend` provides optional dependency-free semantic retrieval for demos and tests.
- `CompositeMemoryBackend` combines two memory backends and supports conflict-safe replace/merge flows.
- `LLMMemoryExtractor` can use an LLM for semantic memory extraction while preserving sensitive-data filtering.
- `OpenAPIToolRegistry` can generate tools from JSON/YAML OpenAPI specs.
- `MCPToolRegistry` provides a minimal stdio MCP adapter for wrapping MCP tools.
- GitHub Actions runs lint, format checks, and tests on Python 3.11.
- Legacy top-level modules such as `miniclaw.agent_loop` remain for older demos; new code should prefer the package layout under `miniclaw.agent`, `miniclaw.tools`, `miniclaw.storage`, and `miniclaw.memory`.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         User Task                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌──────────────────────────────────────────────────────────────┐
│                       Agent Loop                             │
│  ┌─────────┐   ┌─────────┐   ┌────────────┐   ┌──────────┐ │
│  │ Prompt  │──▶│   LLM   │──▶│   Parser   │──▶│ Executor │ │
│  │ Builder │   │         │   │            │   │          │ │
│  └─────────┘   └─────────┘   └────────────┘   └──────────┘ │
│       ▲                                       │             │
│       │            ┌────────────┐             │             │
│       └────────────│  Context   │◀────────────┘             │
│                    │  Manager   │                           │
│                    └────────────┘                           │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │           Tool Registry              │
        │  ┌──────┐ ┌──────┐ ┌──────┐ ┌─────┐│
        │  │ list │ │ read │ │write │ │shell││
        │  │_files│ │_file │ │_file │ │     ││
        │  └──────┘ └──────┘ └──────┘ └─────┘│
        └──────────────────────────────────────┘
                           ▼
        ┌──────────────────────────────────────┐
        │          SQLite Storage              │
        │  sessions · messages · memories      │
        │  · traces                            │
        └──────────────────────────────────────┘
```

## Agent Loop

The core is a **while-loop with a finite-state parser**:

```
while step < max_steps:
    prompt   = build(system, tools, history, task)
    raw      = llm.generate(prompt)
    output   = parser.parse(raw)          # → ToolCall | FinalAnswer | ParseError

    match output:
        FinalAnswer  → return answer
        ToolCall     → executor.run(tool, args) → Observation
                     → append to history → continue
        ParseError   → recovery.handle() → inject repair hint → continue

    if error_count >= max_errors:
        return abort
```

Key design decisions:

- **Parser is strict.** The LLM must return exactly one JSON object with a `type` field. No regex, no fuzzy matching — the model learns the protocol or gets a recovery hint.
- **Errors are observations, not exceptions.** Tool failures, parse errors, and unknown tools are all fed back as context so the model can self-correct.
- **The loop never crashes.** Every failure mode is caught and returned as a structured result.

## Tool Calling Protocol

The LLM communicates tool calls via JSON:

```json
{
    "type": "tool_call",
    "thought": "I need to check the directory contents first.",
    "tool_name": "list_files",
    "arguments": {"path": "."}
}
```

Final answers use the same envelope:

```json
{
    "type": "final_answer",
    "thought": "I've seen the file listing and can now summarize.",
    "answer": "The project contains 12 Python files organized into..."
}
```

Tools are registered as Python classes with a JSON Schema:

```python
class ListFiles(Tool):
    name = "list_files"
    description = "List files and directories at a given path."
    schema = {
        "type": "object",
        "properties": {"path": {"type": "string"}},
        "required": ["path"],
    }

    def run(self, path: str, **kwargs) -> dict:
        ...
```

## Recovery Manager

The agent never crashes on bad output. Each failure type has a targeted recovery strategy:

| Failure | Recovery |
|---------|----------|
| **Invalid JSON** | Extract first `{...}` block via brace counting; if that fails, inject format hint |
| **Unknown tool** | Return list of available tools, remind model to use one |
| **Missing / wrong arguments** | Return the tool's JSON Schema + validation error |
| **Tool execution error** | Return error as observation, suggest retry or final_answer |
| **Consecutive failures** | After N errors, emit a final_answer explaining the failure |

Every recovery message is written to be **machine-readable** — it goes back into the context so the LLM can fix itself on the next iteration.

## Context Compression

When the estimated token count exceeds `max_context_tokens`:

1. **System messages** are always preserved.
2. The **last N messages** (`recent_keep`) are pinned — they represent the current reasoning state.
3. Everything in between is **summarized** into a single `summary` message.

The default summarizer is rule-based (no LLM call required):

```
[Compressed 15 messages]
Roles: {'user': 5, 'assistant': 5, 'tool': 5}
Tools used: get_weather, calculator
Last user: What's the weather in Beijing?
Last assistant: Called get_weather({"city": "Beijing"})
```

To swap in an LLM summarizer:

```python
def llm_summarizer(messages):
    return llm.generate(f"Summarize this conversation:\n{messages}")

ctx = ContextManager(summarizer=llm_summarizer)
```

## Persistent Memory

All state is stored in SQLite via `SQLiteStore`:

| Table | Purpose |
|-------|---------|
| `sessions` | One row per conversation |
| `messages` | Full message history per session |
| `memories` | Key-value long-term memory with importance ranking |
| `traces` | Per-step event log for debugging |

```python
with SQLiteStore(".miniclaw/miniclaw.db") as store:
    sid = store.create_session("Weather task")
    store.save_message(sid, "user", "What's the weather?")
    store.save_memory("user:city", "Beijing", importance=5)
    results = store.search_memories("beijing")
```

## Long-Term Memory with Mem0

LLM APIs are stateless — every request starts from zero. An Agent Harness needs an external memory layer to carry knowledge across sessions.

MiniClaw uses **two storage systems** for different purposes:

| Layer | What it stores | Engine |
| --- | --- | --- |
| **SQLite** | sessions, messages, traces — structured execution logs | `sqlite3` (stdlib) |
| **Mem0** | user preferences, project facts, long-term natural language memories | `mem0ai` (semantic retrieval) |

> **SQLite records what happened; Mem0 remembers what will be useful later.**

### Usage

```bash
# Save a memory
uv run miniclaw memory add "用户偏好中文解释 Agent 底层流程" --user-id michael

# Search memories
uv run miniclaw memory search "用户喜欢什么回答风格？" --user-id michael

# Run with memory enabled
uv run miniclaw run "讲一下 RecoveryManager" --user-id michael --memory

# Run with memory disabled
uv run miniclaw run "task" --no-memory
```

### How it works inside AgentLoop

```text
User Task
    │
    ▼
Memory Search ─── mem0.search(task, user_id) → related memories
    │
    ▼
Memory Injection ─── "## Long-Term Memory\n- 用户偏好中文解释..."
    │
    ▼
LLM Decision ─── model sees memories in context, uses them if relevant
    │
    ▼
Tool Execution → Observation
    │
    ▼
Final Answer
    │
    ▼
Memory Extraction ─── MemoryExtractor.should_remember(task)?
    │
    ▼
Memory Add ─── mem0.add(extracted_text, user_id)
```

Memory search failures never crash the agent. Memory add failures are silently logged.

## Quick Start

```bash
# Install dependencies
uv sync --extra dev

# Run with FakeLLM (no API key needed)
uv run miniclaw run "list files in current directory"

# Run with OpenAI
export OPENAI_API_KEY=sk-...
uv run miniclaw run --llm openai "summarize the project structure"

# Interactive chat
uv run miniclaw chat --llm openai

# View memories and traces
uv run miniclaw memory list
uv run miniclaw trace list

# Export trace as structured JSON
uv run miniclaw trace export --session 1 -o trace.json

# Export trace as HTML
uv run miniclaw trace html --session 1 -o trace_report.html

# Demo: context compression
uv run miniclaw demo context-compression

# Environment health check
uv run miniclaw doctor

# Trace summary for the latest session
uv run miniclaw trace summary

# Replay trace events for the latest session
uv run miniclaw trace replay
```

## Configuration

`miniclaw.toml` is the local project config. CLI flags override config values, and `MINICLAW_*` environment variables override both defaults and config files.

Useful environment overrides:

```bash
MINICLAW_LLM_PROVIDER=openai
MINICLAW_MODEL=gpt-4o-mini
MINICLAW_API_KEY=sk-...
MINICLAW_DB_PATH=.miniclaw/dev.db
MINICLAW_ALLOW_FILE_WRITE=false
MINICLAW_ALLOW_SHELL=false
MINICLAW_SHELL_ALLOWED_PREFIXES=echo,python -m pytest
```

## Demo

### Successful run with recovery

```text
$ uv run miniclaw run -v "list files and describe the project"

🐾 MiniClaw v0.4.0 — session #1
📋 Task: list files and describe the project

✅ Answer (5 steps):

## 项目结构分析
**MiniClaw** 是一个从零实现的轻量级 Agent Harness...
- src/miniclaw/agent/ — Agent 核心
- src/miniclaw/tools/ — 工具系统
...

── Trace ──
  Step 1: 🔧 open_file({'path': 'README.md'})
         ❌ Tool 'open_file' does not exist. Available tools: list_files, read_file, run_shell, write_file.
  Step 2: 🔧 read_file({'path': 'README.md'})
         → {'path': '...', 'content': '# MiniClaw\n\nA lightweight...'}
  Step 3: 🔧 list_files({'path': 'src/miniclaw'})
         → {'entries': [...]}
  Step 4: 🔧 list_files({'path': 'src/miniclaw/agent'})
         → {'entries': [...]}
  Step 5: ✅ final_answer
```

The agent **self-corrects** at Step 1: it called the non-existent `open_file`, received a recovery hint from `RecoveryManager`, and switched to `read_file` at Step 2. The error is logged in the trace but does not crash the loop.

## Tests

```bash
# Run all tests
uv run pytest

# With coverage
uv run pytest --cov=miniclaw --cov-report=term-missing

# Run specific module
uv run pytest tests/test_agent_loop.py -v
```

```text
======================= 662 passed, 1 skipped in ... =======================
```

Pytest is configured to use `.test-tmp` as its base temp directory. If Windows leaves a stale ACL on that folder, delete it and rerun `uv run pytest`.

## Project Structure

```
MiniClaw/
├── main.py                          # Compatibility wrapper for local runs
├── pyproject.toml                   # Project metadata and dependencies
├── requirements.txt
├── miniclaw.toml                    # Example local configuration (gitignored)
├── ROADMAP.md                       # Development plan
├── CODE_WALKTHROUGH.md             # Execution flow walkthrough
│
├── src/miniclaw/
│   ├── cli.py                       # Canonical CLI entry point
│   │
│   ├── agent/                       # Core agent runtime
│   │   ├── loop.py                  # AgentLoop — core while-loop
│   │   ├── async_loop.py            # AsyncAgentLoop variant
│   │   ├── state.py                 # Pydantic models: ToolCall, FinalAnswer
│   │   ├── parser.py                # JSON parser with validation
│   │   ├── executor.py              # ToolExecutor + Observation
│   │   ├── prompts.py               # Prompt builder (system + tools + task + memory)
│   │   ├── context.py               # ContextManager with compression
│   │   ├── recovery.py              # RecoveryManager (5 recovery strategies)
│   │   ├── trace.py                 # StepTrace + TraceLogger, Mermaid/HTML export
│   │   └── config.py                # TOML config loader
│   │
│   ├── tools/                       # Tool system
│   │   ├── base.py                  # Tool abstract base class
│   │   ├── registry.py              # ToolRegistry
│   │   ├── file_tools.py            # list_files, read_file, write_file
│   │   ├── shell_tool.py            # run_shell with safety guardrails
│   │   ├── search_tool.py           # web_search (permission-gated)
│   │   ├── permissions.py           # PermissionPolicy (default-deny)
│   │   ├── audit.py                 # AuditLogger for tool execution
│   │   ├── security.py              # Workspace path resolution
│   │   ├── timeout.py               # Tool timeout/cancellation helpers
│   │   ├── sandbox.py               # Restricted shell execution mode
│   │   ├── openapi_adapter.py       # OpenAPI spec -> tools adapter
│   │   └── mcp_adapter.py           # Minimal stdio MCP adapter
│   │
│   ├── llm/                         # LLM abstraction layer
│   │   ├── base.py                  # BaseLLM abstract interface
│   │   ├── fake.py                  # FakeLLM for testing
│   │   ├── openai_client.py         # OpenAI client with streaming
│   │   └── openai.py                # Legacy OpenAI client
│   │
│   ├── memory/                      # Memory abstraction
│   │   ├── base.py                  # MemoryBackend + NullMemoryBackend
│   │   ├── extractor.py             # Keyword/LLM memory extractors
│   │   ├── manager.py               # Conflict resolution and decay coordinator
│   │   ├── composite.py             # Composite backend
│   │   ├── mem0_store.py            # Mem0MemoryBackend (semantic search)
│   │   └── vector.py                # VectorMemoryBackend (in-memory)
│   │
│   ├── storage/                     # Persistent storage
│   │   ├── sqlite_store.py          # SQLite sessions/messages/memories/traces
│   │   └── memory.py                # Legacy SQLite key-value store
│   │
│   └── multiagent/                  # Multi-agent prototype
│       ├── agents.py                # PlannerAgent, CoderAgent, ReviewerAgent
│       └── coordinator.py           # Coordinator (sequential pipeline)
│
└── tests/                           # 660+ tests across the runtime
```

## Resume Highlights

- **Designed and implemented a complete LLM Agent Runtime from scratch** — agent loop, tool calling protocol, context compression, error recovery, and persistent memory — without relying on any existing agent framework (LangChain, AutoGen, CrewAI).

- **Built a structured tool-calling protocol** with Pydantic-validated JSON schemas, a registry pattern for tool discovery, and a recovery manager that transforms malformed LLM output, unknown tool calls, and execution failures into self-correcting feedback loops.

- **Engineered context management and memory persistence** using a sliding-window compression strategy with rule-based summarization, SQLite-backed storage, optional vector retrieval, trace replay/export, and 660+ passing unit tests across the runtime.

- **Extended with Mem0-based long-term semantic memory** — injects user preferences and project facts into context before each task, extracts high-value memories after completion, and decouples short-term execution state (SQLite) from long-term user knowledge (Mem0) via pluggable MemoryBackend abstraction.

## Future Work

See `ROADMAP.md`.

## License

MIT
