"""Compatibility wrapper for the MiniClaw CLI.

The canonical CLI implementation lives in ``miniclaw.cli`` so the installed
``miniclaw`` console script and ``python main.py`` use the same entry point.
"""

from __future__ import annotations

from miniclaw.cli import build_parser, main

__all__ = ["build_parser", "main"]


if __name__ == "__main__":
    main()
