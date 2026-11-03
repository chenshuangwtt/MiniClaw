# Demo Guide

The offline maintenance demo is the recommended first run:

```powershell
uv run python examples/demo_repository_maintenance.py
```

It copies `examples/repository_maintenance_fixture` to a temporary directory, then has a scripted FakeLLM list files, read source and tests, request a file write, run the fixture test, and emit a trace. It requires no API key and leaves the checked-in fixture unchanged.

For a provider-backed run, configure `OPENAI_API_KEY` outside source control and use native calling:

```powershell
uv run miniclaw run --llm openai --mode native --allow-file-write "Inspect and propose a small fix"
```

Enable `--allow-shell`, `--allow-search`, or `--allow-network` only for the specific run that needs it.

For a paused local run, inspect its run ID and explicitly resume it:

```powershell
uv run miniclaw resume RUN_ID --approve-pending --allow-file-write
```

Runs marked as `in_flight` require manual reconciliation; MiniClaw will not blindly repeat them.
