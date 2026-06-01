"""Tests for cli.py."""

import pytest
from miniclaw.cli import build_parser, main


class TestCLIParser:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc_info:
            build_parser().parse_args(["--version"])
        assert exc_info.value.code == 0

    def test_run_default(self):
        args = build_parser().parse_args(["run"])
        assert args.command == "run"
        assert args.llm == "fake"
        assert args.max_turns == 10

    def test_run_with_openai(self):
        args = build_parser().parse_args(["run", "--llm", "openai", "--model", "gpt-4o"])
        assert args.llm == "openai"
        assert args.model == "gpt-4o"

    def test_no_command_prints_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        assert "MiniClaw" in captured.out or "usage" in captured.out.lower()
