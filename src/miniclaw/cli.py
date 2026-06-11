"""MiniClaw CLI — command-line interface for the agent harness.

Usage::

    # Run a single task
    python main.py run "What files are in the current directory?"

    # Interactive chat
    python main.py chat

    # List saved memories
    python main.py memory list

    # List traces from last session
    python main.py trace list

    # Use OpenAI instead of FakeLLM
    python main.py run --llm openai "Summarize this file"

    # Specify a database file
    python main.py --db .miniclaw/miniclaw.db run "task"
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from miniclaw import __version__


def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the CLI."""
    parser = argparse.ArgumentParser(
        prog="miniclaw",
        description="MiniClaw — a lightweight, self-built Agent Harness.",
    )
    parser.add_argument("--version", action="version", version=f"miniclaw {__version__}")
    parser.add_argument(
        "--config",
        default=None,
        help="Path to config file (default: miniclaw.toml).",
    )
    parser.add_argument(
        "--db",
        default=None,
        help="SQLite database path (overrides config).",
    )

    sub = parser.add_subparsers(dest="command", help="Available commands")

    # --- run ---
    run_p = sub.add_parser("run", help="Run a single task.")
    run_p.add_argument("task", help="The task description.")
    run_p.add_argument(
        "--llm", choices=["fake", "openai"], default=None, help="LLM backend (overrides config)."
    )
    run_p.add_argument("--model", default=None, help="Model name (overrides config).")
    run_p.add_argument("--api-key", default=None, help="OpenAI API key (overrides config).")
    run_p.add_argument("--base-url", default=None, help="Custom API base URL (overrides config).")
    run_p.add_argument(
        "--max-steps", type=int, default=None, help="Max agent steps (overrides config)."
    )
    run_p.add_argument(
        "--max-errors", type=int, default=None, help="Max consecutive errors (overrides config)."
    )
    run_p.add_argument("--verbose", "-v", action="store_true", help="Show trace after run.")
    run_p.add_argument(
        "--stream", action="store_true", default=False, help="Stream LLM output token by token."
    )
    run_p.add_argument(
        "--user-id", default="default", help="User ID for memory (default: default)."
    )
    run_p.add_argument("--memory", action="store_true", default=False, help="Enable Mem0 memory.")
    run_p.add_argument("--no-memory", action="store_true", default=False, help="Disable memory.")
    run_p.add_argument(
        "--allow-file-write", action="store_true", default=False, help="Allow file write tool."
    )
    run_p.add_argument(
        "--allow-shell", action="store_true", default=False, help="Allow shell tool."
    )
    run_p.add_argument(
        "--allow-search", action="store_true", default=False, help="Allow web search tool."
    )

    # --- chat ---
    chat_p = sub.add_parser("chat", help="Interactive chat mode.")
    chat_p.add_argument(
        "--llm", choices=["fake", "openai"], default=None, help="LLM backend (overrides config)."
    )
    chat_p.add_argument("--model", default=None, help="Model name (overrides config).")
    chat_p.add_argument("--api-key", default=None, help="OpenAI API key (overrides config).")
    chat_p.add_argument("--base-url", default=None, help="Custom API base URL (overrides config).")
    chat_p.add_argument(
        "--max-steps", type=int, default=None, help="Max agent steps (overrides config)."
    )

    # --- memory ---
    mem_p = sub.add_parser("memory", help="Manage long-term memory.")
    mem_sub = mem_p.add_subparsers(dest="memory_action")
    mem_sub.add_parser("list", help="List all saved memories (SQLite).")
    add_p = mem_sub.add_parser("add", help="Save a memory via Mem0.")
    add_p.add_argument("text", help="Text to remember.")
    add_p.add_argument("--user-id", default="default", help="User ID.")
    search_p = mem_sub.add_parser("search", help="Search memories via Mem0.")
    search_p.add_argument("query", help="Search query.")
    search_p.add_argument("--user-id", default="default", help="User ID.")
    search_p.add_argument("--limit", type=int, default=5, help="Max results.")

    # --- trace ---
    trace_p = sub.add_parser("trace", help="View execution traces.")
    trace_sub = trace_p.add_subparsers(dest="trace_action")
    trace_sub.add_parser("list", help="List traces from the latest session.")
    summary_p = trace_sub.add_parser("summary", help="Summarize traces from a session.")
    summary_p.add_argument(
        "--session", type=int, default=None, help="Session ID (default: latest)."
    )
    replay_p = trace_sub.add_parser("replay", help="Replay trace events from a session.")
    replay_p.add_argument("--session", type=int, default=None, help="Session ID (default: latest).")
    export_p = trace_sub.add_parser("export", help="Export traces as JSON.")
    export_p.add_argument("--session", type=int, default=None, help="Session ID (default: latest).")
    export_p.add_argument(
        "--output", "-o", default=None, help="Output file path (default: stdout)."
    )
    mermaid_p = trace_sub.add_parser("mermaid", help="Export trace as Mermaid flowchart.")
    mermaid_p.add_argument(
        "--session", type=int, default=None, help="Session ID (default: latest)."
    )
    mermaid_p.add_argument(
        "--output", "-o", default=None, help="Output file path (default: stdout)."
    )
    html_p = trace_sub.add_parser("html", help="Export trace as HTML report.")
    html_p.add_argument("--session", type=int, default=None, help="Session ID (default: latest).")
    html_p.add_argument(
        "--output", "-o", default=None, help="Output file path (default: trace_report.html)."
    )

    # --- demo ---
    demo_p = sub.add_parser("demo", help="Run built-in demos.")
    demo_sub = demo_p.add_subparsers(dest="demo_action")
    demo_sub.add_parser("context-compression", help="Demonstrate context compression.")

    sub.add_parser("doctor", help="Check local environment and MiniClaw configuration.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    # Load config from file, then override with CLI args
    from miniclaw.agent.config import load_config

    config = load_config(args.config)
    _apply_cli_overrides(config, args)

    if args.command == "run":
        _cmd_run(args, config)
    elif args.command == "chat":
        _cmd_chat(args, config)
    elif args.command == "memory":
        _cmd_memory(args, config)
    elif args.command == "trace":
        _cmd_trace(args, config)
    elif args.command == "demo":
        if args.demo_action == "context-compression":
            _cmd_demo_context_compression()
        else:
            print("Available demos: context-compression")
    elif args.command == "doctor":
        _cmd_doctor(config)
    else:
        parser.print_help()


def _apply_cli_overrides(config, args) -> None:
    """Apply CLI arguments that override config file values."""
    if hasattr(args, "llm") and args.llm is not None:
        config.llm.provider = args.llm
    if hasattr(args, "model") and args.model is not None:
        config.llm.model = args.model
    if hasattr(args, "api_key") and args.api_key is not None:
        config.llm.api_key = args.api_key
    if hasattr(args, "base_url") and args.base_url is not None:
        config.llm.base_url = args.base_url
    if hasattr(args, "max_steps") and args.max_steps is not None:
        config.agent.max_steps = args.max_steps
    if hasattr(args, "max_errors") and args.max_errors is not None:
        config.agent.max_errors = args.max_errors
    if hasattr(args, "db") and args.db is not None:
        config.storage.db_path = args.db


# ------------------------------------------------------------------
# Commands
# ------------------------------------------------------------------


def _cmd_run(args: argparse.Namespace, config) -> None:
    """Run a single task."""
    from miniclaw.agent.loop import AgentLoop
    from miniclaw.memory.base import NullMemoryBackend
    from miniclaw.tools.audit import AuditLogger
    from miniclaw.storage.sqlite_store import SQLiteStore

    llm = _create_llm_from_config(config)
    permission_policy = _permission_policy_from_config(config, args)
    audit_logger = AuditLogger()
    registry = _register_default_tools(config, args)
    memory_backend = _resolve_memory_backend(args)

    with SQLiteStore(config.storage.db_path) as store:
        session_id = store.create_session(title=args.task[:80])

        mem_label = "Mem0" if getattr(args, "memory", False) else "off"
        print(f"🐾 MiniClaw v{__version__} — session #{session_id}  memory={mem_label}")
        print(f"📋 Task: {args.task}\n")

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            max_steps=config.agent.max_steps,
            max_errors=config.agent.max_errors,
            memory_backend=memory_backend,
            permission_policy=permission_policy,
            audit_logger=audit_logger,
        )

        # Show injected memories in verbose mode
        if args.verbose and not isinstance(memory_backend, NullMemoryBackend):
            try:
                mems = memory_backend.search(args.task, user_id=args.user_id, limit=5)
                if mems:
                    print(f"🧠 Injected {len(mems)} memories:")
                    for m in mems:
                        print(f"   - {m[:80]}")
                    print()
            except Exception:
                pass

        result = agent.run(args.task, user_id=args.user_id)

        # Save to storage
        for step in result.trace.steps:
            store.save_trace(session_id, step.step, step.model_dump_json())

        if result.success:
            store.save_message(session_id, "user", args.task)
            store.save_message(session_id, "assistant", result.answer)
            if getattr(args, "stream", False):
                print(f"✅ Answer ({result.steps_taken} steps):")
                print()
                _stream_print(result.answer)
                print()
            else:
                print(f"✅ Answer ({result.steps_taken} steps):")
                print(f"\n{result.answer}\n")
        else:
            print(f"❌ Failed after {result.steps_taken} steps:")
            print(f"\n{result.error}\n")

        if args.verbose:
            _print_trace(result.trace.steps)
            _print_audit(audit_logger.events())


def _cmd_chat(args: argparse.Namespace, config) -> None:
    """Interactive chat mode."""
    from miniclaw.agent.loop import AgentLoop
    from miniclaw.tools.audit import AuditLogger
    from miniclaw.storage.sqlite_store import SQLiteStore

    llm = _create_llm_from_config(config)
    permission_policy = _permission_policy_from_config(config)
    audit_logger = AuditLogger()
    registry = _register_default_tools(config)

    with SQLiteStore(config.storage.db_path) as store:
        session_id = store.create_session(title="Interactive chat")
        print(f"🐾 MiniClaw v{__version__} — interactive mode (session #{session_id})")
        print("Type your task, or 'exit' to quit.\n")

        agent = AgentLoop(
            llm=llm,
            registry=registry,
            max_steps=config.agent.max_steps,
            permission_policy=permission_policy,
            audit_logger=audit_logger,
        )

        try:
            while True:
                try:
                    task = input("You> ").strip()
                except EOFError:
                    break

                if not task:
                    continue
                if task.lower() in ("exit", "quit", "q"):
                    break

                result = agent.run(task)

                # Save to storage
                for step in result.trace.steps:
                    store.save_trace(session_id, step.step, step.model_dump_json())
                store.save_message(session_id, "user", task)

                if result.success:
                    store.save_message(session_id, "assistant", result.answer)
                    print(f"\n🤖 {result.answer}\n")
                else:
                    print(f"\n❌ {result.error}\n")

        except KeyboardInterrupt:
            pass

        print("\nGoodbye! 🐾")


def _cmd_memory(args: argparse.Namespace, config) -> None:
    """Memory subcommands."""
    from miniclaw.storage.sqlite_store import SQLiteStore

    if args.memory_action == "list":
        with SQLiteStore(config.storage.db_path) as store:
            memories = store.list_memories()
            if not memories:
                print("No memories saved.")
                return
            print(f"📚 Memories ({len(memories)}):\n")
            for m in memories:
                importance = "⭐" * m["importance"]
                print(f"  [{m['key']}] = {m['value']}")
                print(f"    Importance: {importance}  Updated: {m['updated_at']}\n")

    elif args.memory_action == "add":
        backend = _create_mem0_backend()
        if backend is None:
            return
        backend.add(args.text, user_id=args.user_id)
        print(f"✅ Memory saved for user '{args.user_id}':")
        print(f"   {args.text}")

    elif args.memory_action == "search":
        backend = _create_mem0_backend()
        if backend is None:
            return
        results = backend.search(args.query, user_id=args.user_id, limit=args.limit)
        if not results:
            print(f"No memories found for '{args.query}'.")
            return
        print(f"🔍 Found {len(results)} memories for '{args.query}':\n")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r}")

    else:
        print("Usage: miniclaw memory list")
        print('       miniclaw memory add "text" [--user-id ID]')
        print('       miniclaw memory search "query" [--user-id ID] [--limit N]')


def _cmd_trace(args: argparse.Namespace, config) -> None:
    """Trace subcommands."""
    from miniclaw.storage.sqlite_store import SQLiteStore

    if args.trace_action == "list":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, None)
            if session_id is None:
                return
            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return
            print(f"🔍 Traces for session #{session_id} ({len(traces)} steps):\n")
            _print_trace_from_dicts(traces)

    elif args.trace_action == "summary":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, args.session)
            if session_id is None:
                return

            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return

            conn = store._get_conn()
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = session_row["title"] if session_row else ""
            summary = _summarize_trace_dicts(traces)

            print(f"Trace Summary for session #{session_id}")
            if title:
                print(f"Task: {title}")
            print(f"Steps: {summary['steps']}")
            print(f"Result: {summary['result']}")
            print(f"Errors: {summary['errors']}")
            if summary["tools"]:
                print("Tools:")
                for name, count in summary["tools"].items():
                    print(f"  - {name}: {count}")
            else:
                print("Tools: none")

    elif args.trace_action == "replay":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, args.session)
            if session_id is None:
                return

            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return

            conn = store._get_conn()
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = session_row["title"] if session_row else ""

            print(f"Trace Replay for session #{session_id}")
            if title:
                print(f"Task: {title}")
            print()
            _print_trace_replay(traces)

    elif args.trace_action == "export":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, args.session)
            if session_id is None:
                return

            # Get session title
            conn = store._get_conn()
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = session_row["title"] if session_row else ""

            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return

            # Build export document
            steps = []
            for t in traces:
                try:
                    event = json.loads(t["event_json"])
                except (json.JSONDecodeError, KeyError):
                    event = {}

                step_doc = {
                    "step": t["step"],
                    "type": event.get("parsed_action", "unknown"),
                    "timestamp": t["created_at"],
                }
                if event.get("tool_name"):
                    step_doc["tool_name"] = event["tool_name"]
                if event.get("arguments"):
                    step_doc["arguments"] = event["arguments"]
                if event.get("observation") is not None:
                    step_doc["observation"] = event["observation"]
                if event.get("error"):
                    step_doc["error"] = event["error"]
                if event.get("model_output"):
                    step_doc["model_output"] = event["model_output"]

                steps.append(step_doc)

            export_doc = {
                "session_id": session_id,
                "task": title,
                "total_steps": len(steps),
                "steps": steps,
            }

            output = json.dumps(export_doc, indent=2, ensure_ascii=False)

            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(output, encoding="utf-8")
                print(f"Exported {len(steps)} steps to {args.output}")
            else:
                print(output)

    elif args.trace_action == "mermaid":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, args.session)
            if session_id is None:
                return

            conn = store._get_conn()
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = session_row["title"] if session_row else f"Session #{session_id}"

            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return

            # Build a TraceLogger from stored traces and generate Mermaid
            from miniclaw.agent.trace import TraceLogger

            trace = TraceLogger()
            for t in traces:
                try:
                    event = json.loads(t["event_json"])
                except (json.JSONDecodeError, KeyError):
                    event = {}
                trace.log_step(
                    step=t["step"],
                    model_output=event.get("model_output", ""),
                    parsed_action=event.get("parsed_action", ""),
                    tool_name=event.get("tool_name"),
                    arguments=event.get("arguments"),
                    observation=event.get("observation"),
                    error=event.get("error"),
                )

            mermaid = trace.to_mermaid(title)
            if args.output:
                Path(args.output).parent.mkdir(parents=True, exist_ok=True)
                Path(args.output).write_text(f"```mermaid\n{mermaid}\n```\n", encoding="utf-8")
                print(f"Mermaid flowchart saved to {args.output}")
            else:
                print(mermaid)

    elif args.trace_action == "html":
        with SQLiteStore(config.storage.db_path) as store:
            session_id = _resolve_session_id(store, args.session)
            if session_id is None:
                return

            conn = store._get_conn()
            session_row = conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)
            ).fetchone()
            title = session_row["title"] if session_row else f"Session #{session_id}"

            traces = store.list_traces(session_id)
            if not traces:
                print(f"No traces for session #{session_id}.")
                return

            from miniclaw.agent.trace import TraceLogger

            trace = TraceLogger()
            for t in traces:
                try:
                    event = json.loads(t["event_json"])
                except (json.JSONDecodeError, KeyError):
                    event = {}
                trace.log_step(
                    step=t["step"],
                    model_output=event.get("model_output", ""),
                    parsed_action=event.get("parsed_action", ""),
                    tool_name=event.get("tool_name"),
                    arguments=event.get("arguments"),
                    observation=event.get("observation"),
                    error=event.get("error"),
                )

            output_path = args.output or "trace_report.html"
            trace.export_html(output_path, title=title)
            print(f"HTML report saved to {output_path}")

    else:
        print("Usage: miniclaw trace list")
        print("       miniclaw trace summary [--session N]")
        print("       miniclaw trace replay [--session N]")
        print("       miniclaw trace export [--session N] [-o file.json]")
        print("       miniclaw trace mermaid [--session N] [-o file.md]")
        print("       miniclaw trace html [--session N] [-o file.html]")


def _cmd_doctor(config) -> None:
    """Print a compact local environment health report."""
    print(f"MiniClaw doctor - v{__version__}")
    print(f"Python: {sys.version.split()[0]} ({sys.executable})")
    print(f"CWD: {Path.cwd()}")

    config_path = Path("miniclaw.toml")
    print(f"Config: {'found' if config_path.exists() else 'not found'} ({config_path})")
    print(f"LLM provider: {config.llm.provider}")
    print(f"Model: {config.llm.model}")
    print(f"Database: {config.storage.db_path}")
    print(f"File writes: {'enabled' if config.tools.allow_file_write else 'disabled'}")
    print(f"Shell: {'enabled' if config.tools.allow_shell else 'disabled'}")
    print(f"Web search: {'enabled' if config.tools.allow_search else 'disabled'}")
    prefixes = config.tools.shell_allowed_prefixes
    print(f"Shell prefixes: {', '.join(prefixes) if prefixes else 'any safe command'}")

    api_key = config.llm.api_key or os.getenv("OPENAI_API_KEY")
    print(f"OPENAI_API_KEY: {'set' if api_key else 'not set'}")

    try:
        import openai  # noqa: F401

        print("openai: installed")
    except Exception:
        print("openai: missing")

    try:
        import tiktoken  # noqa: F401

        print("tiktoken: installed")
    except Exception:
        print("tiktoken: missing")

    try:
        import mem0  # noqa: F401

        print("mem0ai: installed")
    except Exception:
        print("mem0ai: missing or unavailable")

    db_parent = Path(config.storage.db_path).expanduser().parent
    if str(db_parent) in ("", "."):
        db_parent = Path.cwd()
    print(f"DB directory writable: {'yes' if os.access(db_parent, os.W_OK) else 'no'}")


def _cmd_demo_context_compression() -> None:
    """Demonstrate ContextManager compression with visual output."""
    from miniclaw.agent.context import ContextManager

    max_tokens = 200
    recent_keep = 6
    ctx = ContextManager(max_context_tokens=max_tokens, recent_keep=recent_keep)

    print("🐾 MiniClaw — Demo: Context Compression")
    print("=" * 55)
    print(f"  max_context_tokens = {max_tokens}")
    print(f"  recent_keep        = {recent_keep}")
    print()

    # --- Simulate 15 rounds of tool-calling conversation ---
    print("Simulating 15 rounds of tool-calling conversation...")
    for i in range(1, 16):
        ctx.add_message("user", f"请帮我分析项目中 module_{i} 的代码结构和实现细节，给出改进建议。")
        ctx.add_message("assistant", f"好的，我来分析 module_{i}。让我先读取源码。")
        ctx.add_observation(
            "read_file",
            {"path": f"src/module_{i}.py"},
            f"这是 module_{i} 的完整代码，包含了 {i * 120 + 50} 行 Python 代码。"
            f"主要实现了数据处理管道、错误重试机制和配置管理。",
        )

    # --- Before compression ---
    tokens_before = ctx.estimate_tokens()
    count_before = ctx.message_count
    need_compress = ctx.should_compress()

    print()
    print("Before compression:")
    print(f"  - messages:          {count_before}")
    print(f"  - estimated tokens:  {tokens_before}")
    print(f"  - should_compress:   {need_compress}")
    print()

    # --- Trigger compression ---
    if need_compress:
        print("Compression triggered.")
        ctx.compress()
    else:
        print("No compression needed (within budget).")

    # --- After compression ---
    tokens_after = ctx.estimate_tokens()
    count_after = ctx.message_count

    print()
    print("After compression:")
    print(f"  - messages:          {count_after}")
    print(f"  - estimated tokens:  {tokens_after}")

    # --- Show summary content ---
    msgs = ctx.get_messages()
    summary_msgs = [m for m in msgs if m.get("role") == "summary"]
    if summary_msgs:
        print()
        print("Summary content:")
        for line in summary_msgs[0]["content"].split("\n"):
            print(f"  {line}")

    # --- Show message structure ---
    print()
    print("Message structure after compression:")
    for i, m in enumerate(msgs):
        role = m["role"]
        content = m["content"]
        if len(content) > 70:
            content = content[:70] + "..."
        print(f"  [{i}] {role:10s} {content}")

    print()
    print("=" * 55)
    print(f"Done. {count_before} messages ({tokens_before} tokens)")
    print(f"  → {count_after} messages ({tokens_after} tokens)")
    reduction = (1 - tokens_after / tokens_before) * 100 if tokens_before else 0
    print(f"  → {reduction:.0f}% token reduction")


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------


def _resolve_session_id(store, session_id: int | None) -> int | None:
    """Resolve session ID: use explicit value or find the latest."""
    if session_id is not None:
        return session_id
    conn = store._get_conn()
    row = conn.execute("SELECT id FROM sessions ORDER BY id DESC LIMIT 1").fetchone()
    if not row:
        print("No sessions found.")
        return None
    return row["id"]


def _resolve_memory_backend(args):
    """Resolve which memory backend to use based on CLI flags."""
    from miniclaw.memory.base import NullMemoryBackend

    if getattr(args, "no_memory", False):
        return NullMemoryBackend()
    if getattr(args, "memory", False):
        backend = _create_mem0_backend()
        if backend is not None:
            return backend
        print("⚠️  Mem0 unavailable, falling back to no memory.")
        return NullMemoryBackend()
    return NullMemoryBackend()


def _create_mem0_backend():
    """Create a Mem0MemoryBackend, or return None with a friendly error."""
    try:
        from miniclaw.memory.mem0_store import Mem0MemoryBackend
    except Exception:
        print("❌ mem0ai is not installed. Install with: pip install mem0ai")
        return None

    backend = Mem0MemoryBackend()
    if not backend.is_available:
        print("❌ Mem0 initialization failed. Check your configuration.")
        return None
    return backend


def _create_llm_from_config(config):
    """Instantiate the LLM backend from a Config object."""
    if config.llm.provider == "fake":
        from miniclaw.llm.fake import FakeLLM

        return FakeLLM(
            [
                '{"type": "tool_call", "thought": "先读取 README 文件了解项目。", "tool_name": "open_file", "arguments": {"path": "README.md"}}',
                '{"type": "tool_call", "thought": "工具名应该是 read_file，我来修正。", "tool_name": "read_file", "arguments": {"path": "README.md"}}',
                '{"type": "tool_call", "thought": "README 说这是一个 Agent Harness，我来看看源码结构。", "tool_name": "list_files", "arguments": {"path": "src/miniclaw"}}',
                '{"type": "tool_call", "thought": "agent/ 是核心模块，看看里面有哪些文件。", "tool_name": "list_files", "arguments": {"path": "src/miniclaw/agent"}}',
                '{"type": "final_answer", "thought": "我已经了解了项目结构，现在给出分析。", "answer": "## 项目结构分析\\n\\n**MiniClaw** 是一个从零实现的轻量级 Agent Harness，不依赖 LangChain/AutoGen/CrewAI。\\n\\n### 核心模块\\n\\n- **src/miniclaw/agent/** — Agent 核心：loop（主循环）、parser（JSON 解析）、executor（工具执行）、recovery（错误恢复）、context（上下文压缩）、trace（追踪日志）\\n- **src/miniclaw/tools/** — 工具系统：base（抽象基类）、registry（注册中心）、file_tools（文件操作）、shell_tool（Shell 执行，带安全过滤）\\n- **src/miniclaw/llm/** — LLM 抽象层：base（统一接口）、fake（测试替身）、openai_client（OpenAI 兼容客户端）\\n- **src/miniclaw/storage/** — SQLite 持久化：sessions、messages、memories、traces 四张表\\n\\n### 错误恢复演示\\n\\nStep 1 调用了不存在的 `open_file` 工具，RecoveryManager 返回可用工具列表后，Agent 在 Step 2 自动修正为 `read_file`。这展示了 Agent 的自纠正能力。\\n\\n### 设计亮点\\n\\n- Agent Loop 采用 while + 有限状态机设计\\n- RecoveryManager 实现了 5 种错误恢复策略\\n- ContextManager 支持自动上下文压缩\\n- 工具调用协议使用 Pydantic 校验的 JSON Schema"}',
            ]
        )
    elif config.llm.provider == "openai":
        from miniclaw.llm.openai_client import OpenAIClient

        return OpenAIClient(
            model=config.llm.model,
            api_key=config.llm.api_key,
            base_url=config.llm.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {config.llm.provider}")


def _permission_policy_from_config(config, args=None):
    """Build the tool permission policy from config, with CLI overrides."""
    from miniclaw.tools.permissions import PermissionPolicy

    policy = PermissionPolicy(
        allow_file_write=config.tools.allow_file_write,
        allow_shell=config.tools.allow_shell,
        allow_search=config.tools.allow_search,
        shell_allowed_prefixes=config.tools.shell_allowed_prefixes,
    )

    # CLI flags override config (explicit opt-in)
    if args is not None:
        if getattr(args, "allow_file_write", False):
            policy.allow_file_write = True
        if getattr(args, "allow_shell", False):
            policy.allow_shell = True
        if getattr(args, "allow_search", False):
            policy.allow_search = True

    return policy


def _register_default_tools(config=None, args=None):
    """Register the default set of tools."""
    from miniclaw.tools.file_tools import ListFiles, ReadFile, WriteFile
    from miniclaw.tools.registry import ToolRegistry
    from miniclaw.tools.search_tool import WebSearch
    from miniclaw.tools.shell_tool import RunShell

    registry = ToolRegistry()
    workspace_root = Path.cwd()
    allow_file_write = True if config is None else config.tools.allow_file_write
    allow_shell = True if config is None else config.tools.allow_shell
    allow_search = False if config is None else config.tools.allow_search
    shell_allowed_prefixes = None if config is None else config.tools.shell_allowed_prefixes

    if args is not None:
        if getattr(args, "allow_file_write", False):
            allow_file_write = True
        if getattr(args, "allow_shell", False):
            allow_shell = True
        if getattr(args, "allow_search", False):
            allow_search = True

    registry.register(ListFiles(workspace_root=workspace_root))
    registry.register(ReadFile(workspace_root=workspace_root))
    registry.register(WriteFile(workspace_root=workspace_root, allow_write=allow_file_write))
    registry.register(RunShell(allow_shell=allow_shell, allowed_prefixes=shell_allowed_prefixes))
    if allow_search:
        registry.register(WebSearch())
    return registry


def _stream_print(text: str) -> None:
    """Print an already computed answer incrementally without another LLM call."""
    for char in text:
        print(char, end="", flush=True)


def _print_trace(steps) -> None:
    """Pretty-print trace steps."""
    print("── Trace ──")
    for s in steps:
        action = s.parsed_action
        if action == "tool_call":
            if s.error:
                print(f"  Step {s.step}: 🔧 {s.tool_name}({s.arguments})")
                print(f"         ❌ {s.error[:120]}")
            else:
                print(f"  Step {s.step}: 🔧 {s.tool_name}({s.arguments})")
                if s.observation:
                    obs_str = str(s.observation)[:100]
                    print(f"         → {obs_str}")
        elif action == "final_answer":
            print(f"  Step {s.step}: ✅ final_answer")
        elif action == "parse_error":
            print(f"  Step {s.step}: ⚠️  parse error: {s.error}")
        else:
            print(f"  Step {s.step}: {action}")
    print()


def _print_trace_from_dicts(traces: list[dict]) -> None:
    """Pretty-print trace dicts from the database."""
    for t in traces:
        try:
            event = json.loads(t["event_json"])
        except (json.JSONDecodeError, KeyError):
            event = {}

        action = event.get("parsed_action", "?")
        step = t["step"]

        if action == "tool_call":
            name = event.get("tool_name", "?")
            args = event.get("arguments", {})
            print(f"  Step {step}: 🔧 {name}({args})")
        elif action == "final_answer":
            print(f"  Step {step}: ✅ final_answer")
        elif action == "parse_error":
            print(f"  Step {step}: ⚠️  parse error")
        else:
            print(f"  Step {step}: {action}")
    print()


def _print_trace_replay(traces: list[dict]) -> None:
    """Print a chronological replay of stored trace events."""
    for t in traces:
        try:
            event = json.loads(t["event_json"])
        except (json.JSONDecodeError, KeyError):
            event = {}

        step = t["step"]
        action = event.get("parsed_action", "unknown")
        print(f"Step {step}: {action}")

        if event.get("model_output"):
            model_output = str(event["model_output"]).replace("\n", "\\n")
            print(f"  model: {model_output[:160]}")

        if event.get("tool_name"):
            print(f"  tool: {event['tool_name']}({event.get('arguments', {})})")

        if event.get("error"):
            print(f"  error: {event['error']}")
        elif event.get("observation") is not None:
            observation = str(event["observation"]).replace("\n", "\\n")
            print(f"  observation: {observation[:160]}")

    print()


def _print_audit(events) -> None:
    """Pretty-print in-memory tool audit events."""
    if not events:
        return

    print("── Tool Audit ──")
    for event in events:
        status = "allowed" if event.allowed else "blocked"
        result = "success" if event.success else "failed"
        print(f"  {event.tool_name}: {status}, {result}")
        if event.error:
            print(f"    error: {event.error[:120]}")
    print()


def _summarize_trace_dicts(traces: list[dict]) -> dict[str, object]:
    """Return aggregate stats for trace rows from SQLiteStore."""
    tools: dict[str, int] = {}
    errors = 0
    result = "unknown"

    for row in traces:
        try:
            event = json.loads(row["event_json"])
        except (json.JSONDecodeError, KeyError, TypeError):
            event = {}

        action = event.get("parsed_action", "unknown")
        if action == "final_answer":
            result = "success"
        elif action in {"parse_error", "llm_error"}:
            errors += 1
            if result == "unknown":
                result = "failed"

        if event.get("error"):
            errors += 1
            if result == "unknown":
                result = "failed"

        tool_name = event.get("tool_name")
        if tool_name:
            tools[tool_name] = tools.get(tool_name, 0) + 1

    return {
        "steps": len(traces),
        "tools": tools,
        "errors": errors,
        "result": result,
    }


if __name__ == "__main__":
    main()
