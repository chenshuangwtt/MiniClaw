# Architecture

MiniClaw keeps the runtime small enough to inspect end to end.

```text
CLI / Python API -> AgentEngine -> LLM protocol -> ToolExecutor -> ToolRegistry
                       |                |              |
                       |                |              +-> PermissionPolicy / AuditLogger
                       |                +-> portable JSON or native provider tool calls
                       +-> ContextManager / MemoryBackend / TraceLogger / SQLiteStore
```

`AgentEngine` owns the execution semantics. `AgentLoop` calls it with `asyncio.run`; `AsyncAgentLoop` awaits it. This prevents divergent sync and async behavior.

In native mode, the engine appends the provider assistant message containing every requested tool call, executes each call, then appends one `role=tool` message per call with its matching `tool_call_id`. Portable mode keeps JSON-in-text for offline providers and tests.

Runtime records use SQLite: sessions contain user-facing history and traces; runs contain lifecycle status; checkpoints capture serializable context and observations; `agent_memories` are isolated by user and project. A checkpoint is a recovery record, not permission to replay a write or external side effect.
