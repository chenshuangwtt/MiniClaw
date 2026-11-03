# Code Walkthrough

Start at `src/miniclaw/cli.py`, which constructs an LLM, registry, permission policy, SQLite session, and `AgentLoop`.

`src/miniclaw/agent/engine.py` contains the loop. `_run_portable` parses JSON envelopes; `_run_native` handles provider responses and all native tool calls. Both use `_execute`, `ToolExecutor`, traces, context, and the same final-result rules.

`src/miniclaw/agent/executor.py` is the tool trust boundary. It checks the registry, schema, capability policy, timeout, and result status before yielding an `Observation`.

`src/miniclaw/storage/sqlite_store.py` owns durable data. Runs and checkpoints are separate from session messages so diagnostic replay remains read-only.
