# MiniClaw Development Notes

MiniClaw is a local, educational Python Agent Harness. It deliberately avoids LangChain, AutoGen, and CrewAI.

- Use Python 3.11+ with `uv`; run tests through `uv run pytest`.
- Keep runtime code under `src/miniclaw` and tests under `tests`.
- `AgentEngine` is shared by the sync and async public APIs.
- Portable mode accepts JSON-in-text. Native mode follows OpenAI-compatible `tool_calls` and `tool_call_id` association rules.
- Tools are capability-classified. File writes, shell commands, network access, and external side effects require explicit permission.
- Do not describe `SandboxExecutor` as a security boundary. It is process-level best effort only.
- SQLite checkpoints are not authorization to replay a side-effecting tool.
