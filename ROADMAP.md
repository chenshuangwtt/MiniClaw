# MiniClaw Roadmap

## v1.0.0 - Local Agent Runtime

Status: complete for local, educational use.

- Unified async-first engine with sync and async facades
- Portable JSON and OpenAI-compatible native Tool Calling
- Multi-call execution with tool-call ID association
- Schema validation, timeout/error observations, capability-based policy, and audit events
- Workspace-bound file tools; file write, shell, network, and external-side-effect capabilities default to deny
- SQLite session, trace, run, checkpoint, and scoped long-term-memory persistence
- Offline repository-maintenance demonstration and GitHub Actions CI

## Explicit Limits

- This is not a security sandbox. `SandboxExecutor` narrows environment, paths, and command patterns, but it cannot provide OS/container isolation.
- Checkpoints resume cancelled context and explicit approvals. A checkpoint marked `in_flight` is never retried automatically and requires manual reconciliation.
- Native-provider pending tool calls and arbitrary in-process thread cancellation require manual reconciliation/process isolation.
- Trace replay is read-only presentation, not execution resume.
- Provider streaming exists at `BaseLLM.stream()` / `OpenAIClient.stream()`; the CLI does not pretend a completed answer is live streaming.

## Not Planned

- Agent frameworks, web UI, chat channels, distributed execution, or hidden autonomous background work
- Automatic destructive actions or bypasses for the permission policy
