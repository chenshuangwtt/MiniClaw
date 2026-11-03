# MiniClaw v1.0 Code Walkthrough

## 1. Entry Point

`uv run miniclaw run ...` enters `src/miniclaw/cli.py`. The CLI creates an LLM provider, tool registry, permission policy, audit logger, SQLite session, and `AgentLoop`. OpenAI runs default to native tool calling; `FakeLLM` runs default to portable JSON.

## 2. One Engine, Two APIs

`AgentLoop.run()` is the synchronous facade and `AsyncAgentLoop.run()` is the asynchronous facade. Both call `AgentEngine` in `src/miniclaw/agent/engine.py`, so they share context, retry limits, traces, cancellation, lifecycle status, permission handling, and result semantics.

```text
task -> AgentEngine -> LLM -> parse/provider tool_calls -> ToolExecutor
                                                ^                |
                                                |                v
                                          Context/trace <- Observation
```

## 3. Tool Protocols

Portable mode expects a JSON envelope such as:

```json
{"type":"tool_call","tool_name":"read_file","arguments":{"path":"README.md"}}
```

Native mode consumes the provider's `tool_calls` array. For every call, MiniClaw keeps the assistant message, executes the tool, and appends a `role=tool` message with the same `tool_call_id`. Multiple calls in one model response are executed in order.

## 4. The Trust Boundary

`ToolExecutor` validates schema, capability policy, timeout, exceptions, returned error payloads, and command exit codes. Failures are structured `Observation` values. A final answer after an unresolved tool failure produces a failed run rather than a false success.

Capabilities are declared by tools: `read_only`, `file_write`, `shell_execution`, `network_access`, and `external_side_effect`. All but read-only require explicit opt-in.

## 5. Durable Runtime Data

`SQLiteStore` saves sessions, messages, memories, traces, runs, and checkpoints. A checkpoint captures recovery state and approval context. It does not automatically re-run side-effecting tools; trace replay is only a read-only visualization.

## 6. Inspect It Offline

```powershell
uv run python examples/demo_repository_maintenance.py
```

The demo uses `FakeLLM` and a temporary copy of a broken fixture. It lists files, reads source and test files, writes a repair under explicit policy, runs the fixture test, and prints the trace.
