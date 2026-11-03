# Security And Reliability

MiniClaw is local-first and conservative by default. `PermissionPolicy` denies file writes, shell execution, network access, and external side effects until a caller explicitly enables them. File tools resolve relative paths beneath their configured workspace root and reject traversal outside it.

The default runtime memory backend is SQLite. `MemoryManager` writes only extractor-selected fragments, applies its configured conflict strategy, and rejects common API-key, password, token, card, and private-key patterns before persistence. Its records are scoped by `user_id` and workspace-derived project ID; this is logical isolation inside one local database, not encryption or multi-tenant authorization.

`ToolExecutor` validates tool arguments before invocation and converts permission failures, schema failures, timeouts, exceptions, error payloads, and non-zero command exits into failed `Observation` values. A later model final answer cannot change an unresolved tool failure into a successful run.

`RunShell` has a blocklist and optional command-prefix allowlist. `SandboxExecutor` provides best-effort process restrictions such as allowed working paths, environment sanitization, and timeouts. It is **not** an OS security boundary: do not run untrusted commands or credentials through it. Use a real container/VM/OS sandbox for hostile workloads.

`miniclaw resume RUN_ID` restores completed context after an interruption. It refuses to replay a checkpoint marked `in_flight`, because an external operation may already have succeeded without a saved result. A checkpoint waiting for approval runs only with `--approve-pending` and the corresponding capability flag, for example `--allow-file-write`.

Cancellation is cooperative between model/tool steps. A timeout returns control to the engine, but an arbitrary Python thread cannot be forcibly and safely terminated; use subprocess timeouts or real process isolation for commands that need hard cancellation. Native-provider pending tool calls are deliberately left for manual reconciliation rather than replayed.

Avoid committing `.env`, API keys, SQLite runtime data, generated traces, or `.venv`. `.gitignore` covers these local artifacts; review `git status --ignored` before publishing.
