"""CLI entry point for MiniClaw."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from miniclaw import __version__


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="miniclaw",
        description="MiniClaw — a lightweight, self-built Agent Harness.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command")

    # --- run ---
    run_p = sub.add_parser("run", help="Start an interactive agent session.")
    run_p.add_argument(
        "--llm",
        choices=["fake", "openai"],
        default="fake",
        help="LLM backend to use (default: fake).",
    )
    run_p.add_argument("--model", default="gpt-4o-mini", help="Model name (for openai).")
    run_p.add_argument("--api-key", default=None, help="OpenAI API key.")
    run_p.add_argument("--base-url", default=None, help="Custom API base URL.")
    run_p.add_argument("--max-turns", type=int, default=10, help="Max tool-call rounds.")
    run_p.add_argument("--trace-file", default=None, help="Path to write trace JSONL.")
    run_p.add_argument("--db", default=None, help="SQLite DB path for memory.")

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        return

    if args.command == "run":
        _run_interactive(args)


def _run_interactive(args: argparse.Namespace) -> None:
    """Launch the interactive agent loop."""
    from miniclaw.agent_loop import Agent
    from miniclaw.context import ContextManager
    from miniclaw.memory import Memory
    from miniclaw.recovery import RecoveryManager
    from miniclaw.tool_registry import ToolRegistry
    from miniclaw.trace import TraceLogger

    # --- Setup logging ---
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # --- Build components ---
    llm = _create_llm(args)
    tools = ToolRegistry()
    ctx = ContextManager()
    recovery = RecoveryManager()
    trace = TraceLogger(args.trace_file, console=True) if args.trace_file else TraceLogger(console=True)
    memory = Memory(args.db) if args.db else None

    # Register a demo tool
    @tools.register(name="echo", description="Echo back the input text.", parameters={
        "type": "object",
        "properties": {"text": {"type": "string"}},
        "required": ["text"],
    })
    def echo(text: str) -> str:
        return text

    agent = Agent(
        llm=llm,
        tools=tools,
        context=ctx,
        recovery=recovery,
        memory=memory,
        trace=trace,
        max_turns=args.max_turns,
    )

    print(f"🐾 MiniClaw v{__version__} — LLM: {args.llm}")
    print("Type your message (Ctrl+C to quit):\n")

    try:
        while True:
            try:
                user_input = input("You> ").strip()
            except EOFError:
                break
            if not user_input:
                continue
            reply = agent.run(user_input)
            print(f"\n🤖 {reply}\n")
    except KeyboardInterrupt:
        print("\nBye! 🐾")
    finally:
        if memory:
            memory.close()
        if trace:
            trace.flush()


def _create_llm(args: argparse.Namespace):
    """Instantiate the selected LLM backend."""
    if args.llm == "fake":
        from miniclaw.llm.fake import FakeLLM

        return FakeLLM([
            "I'm MiniClaw running on FakeLLM. I would help you, but I'm just a mock! 🐾",
        ])
    elif args.llm == "openai":
        from miniclaw.llm.openai import OpenAILLM

        return OpenAILLM(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM: {args.llm}")


if __name__ == "__main__":
    main()
