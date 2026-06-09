"""Tests for main.py CLI."""

from argparse import Namespace
from contextlib import suppress
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from main import build_parser, main


def _prepare_sandbox_dir(name: str) -> Path:
    """Create an ignored workspace-local directory for CLI side-effect tests."""
    sandbox_dir = Path.cwd() / name
    sandbox_dir.mkdir(exist_ok=True)
    for filename in ("None", "configured.db", "miniclaw.toml"):
        with suppress(OSError):
            (sandbox_dir / filename).unlink()
    return sandbox_dir


# ============================================================
# Parser tests
# ============================================================


class TestParser:
    def test_version(self):
        parser = build_parser()
        with pytest.raises(SystemExit) as exc:
            parser.parse_args(["--version"])
        assert exc.value.code == 0

    def test_run_command(self):
        parser = build_parser()
        args = parser.parse_args(["run", "hello"])
        assert args.command == "run"
        assert args.task == "hello"
        assert args.llm is None  # None means "use config default"
        assert args.max_steps is None

    def test_run_with_openai(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--llm", "openai", "--model", "gpt-4o", "task"])
        assert args.llm == "openai"
        assert args.model == "gpt-4o"

    def test_run_verbose(self):
        parser = build_parser()
        args = parser.parse_args(["run", "-v", "task"])
        assert args.verbose is True

    def test_chat_command(self):
        parser = build_parser()
        args = parser.parse_args(["chat"])
        assert args.command == "chat"
        assert args.llm is None  # None means "use config default"

    def test_memory_list(self):
        parser = build_parser()
        args = parser.parse_args(["memory", "list"])
        assert args.command == "memory"
        assert args.memory_action == "list"

    def test_trace_list(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "list"])
        assert args.command == "trace"
        assert args.trace_action == "list"

    def test_trace_summary(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "summary", "--session", "3"])
        assert args.command == "trace"
        assert args.trace_action == "summary"
        assert args.session == 3

    def test_trace_replay(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "replay", "--session", "3"])
        assert args.command == "trace"
        assert args.trace_action == "replay"
        assert args.session == 3

    def test_doctor_command(self):
        parser = build_parser()
        args = parser.parse_args(["doctor"])
        assert args.command == "doctor"

    def test_db_option(self):
        parser = build_parser()
        args = parser.parse_args(["--db", "test.db", "run", "task"])
        assert args.db == "test.db"

    def test_no_command(self, capsys):
        """No subcommand prints help."""
        main([])
        captured = capsys.readouterr()
        assert "miniclaw" in captured.out.lower() or "usage" in captured.out.lower()


# ============================================================
# run command
# ============================================================


class TestRunCommand:
    def test_run_fake_llm(self, tmp_path, capsys):
        """Run with FakeLLM should complete and print answer."""
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "List files in current directory"])
        captured = capsys.readouterr()
        assert "MiniClaw" in captured.out
        assert "Answer" in captured.out or "Failed" in captured.out

    def test_run_saves_to_db(self, tmp_path, capsys):
        """Run should save trace and messages to SQLite."""
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "test task"])

        with SQLiteStore(db) as store:
            # Should have at least one session
            conn = store._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM sessions").fetchone()
            assert row["cnt"] >= 1

    def test_run_verbose_shows_trace(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "-v", "test task"])
        captured = capsys.readouterr()
        assert "Trace" in captured.out or "Step" in captured.out

    def test_run_custom_max_steps(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "--max-steps", "2", "test task"])
        captured = capsys.readouterr()
        assert "MiniClaw" in captured.out

    def test_run_stream_does_not_call_llm_again(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "--stream", "test task"])
        captured = capsys.readouterr()
        assert "[FakeLLM] No more scripted responses" not in captured.out
        assert "项目结构分析" in captured.out


# ============================================================
# chat command
# ============================================================


class TestChatCommand:
    def test_chat_exits_on_exit(self, tmp_path, capsys, monkeypatch):
        """Chat should exit when user types 'exit'."""

        db = tmp_path / "test.db"
        inputs = iter(["hello", "exit"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        main(["--db", str(db), "chat"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_exits_on_quit(self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "test.db"
        inputs = iter(["quit"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        main(["--db", str(db), "chat"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_exits_on_eof(self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "test.db"

        def raise_eof(_=""):
            raise EOFError

        monkeypatch.setattr("builtins.input", raise_eof)
        main(["--db", str(db), "chat"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_empty_input_skipped(self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "test.db"
        inputs = iter(["", "  ", "exit"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        main(["--db", str(db), "chat"])
        captured = capsys.readouterr()
        assert "Goodbye" in captured.out

    def test_chat_saves_messages(self, tmp_path, capsys, monkeypatch):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        inputs = iter(["test task", "exit"])
        monkeypatch.setattr("builtins.input", lambda _="": next(inputs))

        main(["--db", str(db), "chat"])

        with SQLiteStore(db) as store:
            conn = store._get_conn()
            row = conn.execute("SELECT COUNT(*) as cnt FROM messages").fetchone()
            assert row["cnt"] >= 1  # at least the user message saved


# ============================================================
# memory command
# ============================================================


class TestMemoryCommand:
    def test_memory_uses_configured_db_when_db_not_passed(self, capsys, monkeypatch):
        sandbox_dir = _prepare_sandbox_dir(".sandbox_cli_memory")
        monkeypatch.chdir(sandbox_dir)
        config_path = sandbox_dir / "miniclaw.toml"
        config_path.write_text('[storage]\ndb_path = "configured.db"\n', encoding="utf-8")

        main(["--config", str(config_path), "memory", "list"])

        captured = capsys.readouterr()
        assert "No memories" in captured.out
        assert (sandbox_dir / "configured.db").exists()
        assert not (sandbox_dir / "None").exists()

    def test_memory_list_empty(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "memory", "list"])
        captured = capsys.readouterr()
        assert "No memories" in captured.out

    def test_memory_list_with_data(self, tmp_path, capsys):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            store.save_memory("user:name", "Alice", importance=5)
            store.save_memory("user:city", "Beijing", importance=3)

        main(["--db", str(db), "memory", "list"])
        captured = capsys.readouterr()
        assert "Alice" in captured.out
        assert "Beijing" in captured.out
        assert "Memories" in captured.out

    def test_memory_no_action(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "memory"])
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_memory_add(self, mock_create, capsys):
        mock_backend = MagicMock()
        mock_create.return_value = mock_backend
        main(["memory", "add", "记住我喜欢咖啡", "--user-id", "alice"])
        mock_backend.add.assert_called_once_with("记住我喜欢咖啡", user_id="alice")
        captured = capsys.readouterr()
        assert "saved" in captured.out.lower() or "alice" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_memory_search(self, mock_create, capsys):
        mock_backend = MagicMock()
        mock_backend.search.return_value = ["Alice likes coffee", "Alice prefers Python"]
        mock_create.return_value = mock_backend
        main(["memory", "search", "Alice", "--user-id", "alice"])
        mock_backend.search.assert_called_once_with("Alice", user_id="alice", limit=5)
        captured = capsys.readouterr()
        assert "Alice likes coffee" in captured.out
        assert "Alice prefers Python" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_memory_search_no_results(self, mock_create, capsys):
        mock_backend = MagicMock()
        mock_backend.search.return_value = []
        mock_create.return_value = mock_backend
        main(["memory", "search", "nothing", "--user-id", "alice"])
        captured = capsys.readouterr()
        assert "No memories found" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_memory_add_backend_unavailable(self, mock_create, capsys):
        """When backend is None, the command exits cleanly without crashing."""
        mock_create.return_value = None
        main(["memory", "add", "test"])  # Should not raise


class TestRunMemoryFlags:
    def test_parser_memory_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--memory", "task"])
        assert args.memory is True
        assert args.no_memory is False

    def test_parser_no_memory_flag(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--no-memory", "task"])
        assert args.no_memory is True
        assert args.memory is False

    def test_parser_user_id(self):
        parser = build_parser()
        args = parser.parse_args(["run", "--user-id", "bob", "task"])
        assert args.user_id == "bob"

    def test_parser_user_id_default(self):
        parser = build_parser()
        args = parser.parse_args(["run", "task"])
        assert args.user_id == "default"

    def test_run_no_memory_uses_null(self, tmp_path, capsys):
        """--no-memory should disable memory (NullMemoryBackend)."""
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "--no-memory", "test task"])
        captured = capsys.readouterr()
        assert "memory=off" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_run_with_memory(self, mock_create, tmp_path, capsys):
        """--memory should try to create Mem0 backend."""
        mock_backend = MagicMock()
        mock_backend.search.return_value = ["user likes Python"]
        mock_create.return_value = mock_backend
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "--memory", "-v", "test task"])
        captured = capsys.readouterr()
        assert "memory=Mem0" in captured.out

    @patch("miniclaw.cli._create_mem0_backend")
    def test_run_memory_verbose_shows_injected(self, mock_create, tmp_path, capsys):
        """-v --memory should show injected memories."""
        mock_backend = MagicMock()
        mock_backend.search.return_value = ["Alice likes coffee", "Alice prefers Python"]
        mock_create.return_value = mock_backend
        db = tmp_path / "test.db"
        main(["--db", str(db), "run", "--memory", "-v", "analyze code"])
        captured = capsys.readouterr()
        assert "Injected 2 memories" in captured.out
        assert "Alice likes coffee" in captured.out


# ============================================================
# trace command
# ============================================================


class TestTraceCommand:
    def test_trace_uses_configured_db_when_db_not_passed(self, capsys, monkeypatch):
        sandbox_dir = _prepare_sandbox_dir(".sandbox_cli_trace")
        monkeypatch.chdir(sandbox_dir)
        config_path = sandbox_dir / "miniclaw.toml"
        config_path.write_text('[storage]\ndb_path = "configured.db"\n', encoding="utf-8")

        main(["--config", str(config_path), "trace", "summary"])

        captured = capsys.readouterr()
        assert "No sessions" in captured.out
        assert (sandbox_dir / "configured.db").exists()
        assert not (sandbox_dir / "None").exists()

    def test_trace_list_no_sessions(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "trace", "list"])
        captured = capsys.readouterr()
        assert "No sessions" in captured.out

    def test_trace_list_with_data(self, tmp_path, capsys):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            sid = store.create_session("test")
            store.save_trace(
                sid,
                step=1,
                event_json=json.dumps(
                    {
                        "parsed_action": "tool_call",
                        "tool_name": "list_files",
                        "arguments": {"path": "."},
                    }
                ),
            )
            store.save_trace(
                sid,
                step=2,
                event_json=json.dumps(
                    {
                        "parsed_action": "final_answer",
                    }
                ),
            )

        main(["--db", str(db), "trace", "list"])
        captured = capsys.readouterr()
        assert "Traces" in captured.out
        assert "Step" in captured.out

    def test_trace_no_action(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "trace"])
        captured = capsys.readouterr()
        assert "Usage" in captured.out

    def test_trace_summary_with_data(self, tmp_path, capsys):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            sid = store.create_session("summary task")
            store.save_trace(
                sid,
                step=1,
                event_json=json.dumps(
                    {
                        "parsed_action": "tool_call",
                        "tool_name": "list_files",
                        "arguments": {"path": "."},
                    }
                ),
            )
            store.save_trace(
                sid,
                step=2,
                event_json=json.dumps(
                    {
                        "parsed_action": "tool_call",
                        "tool_name": "read_file",
                        "arguments": {"path": "README.md"},
                        "error": "File not found",
                    }
                ),
            )
            store.save_trace(
                sid,
                step=3,
                event_json=json.dumps(
                    {
                        "parsed_action": "final_answer",
                    }
                ),
            )

        main(["--db", str(db), "trace", "summary"])
        captured = capsys.readouterr()
        assert "Trace Summary" in captured.out
        assert "summary task" in captured.out
        assert "Steps: 3" in captured.out
        assert "Result: success" in captured.out
        assert "Errors: 1" in captured.out
        assert "list_files: 1" in captured.out
        assert "read_file: 1" in captured.out

    def test_trace_summary_no_sessions(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        main(["--db", str(db), "trace", "summary"])
        captured = capsys.readouterr()
        assert "No sessions" in captured.out

    def test_trace_replay_with_data(self, tmp_path, capsys):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            sid = store.create_session("replay task")
            store.save_trace(
                sid,
                step=1,
                event_json=json.dumps(
                    {
                        "parsed_action": "tool_call",
                        "model_output": '{"type":"tool_call"}',
                        "tool_name": "list_files",
                        "arguments": {"path": "."},
                        "observation": {"entries": []},
                    }
                ),
            )
            store.save_trace(
                sid,
                step=2,
                event_json=json.dumps(
                    {
                        "parsed_action": "final_answer",
                        "observation": "done",
                    }
                ),
            )

        main(["--db", str(db), "trace", "replay"])
        captured = capsys.readouterr()
        assert "Trace Replay" in captured.out
        assert "replay task" in captured.out
        assert "Step 1: tool_call" in captured.out
        assert "tool: list_files" in captured.out
        assert "Step 2: final_answer" in captured.out


# ============================================================
# _register_default_tools
# ============================================================


class TestDefaultTools:
    def test_default_tools_registered(self):
        from miniclaw.cli import _register_default_tools

        registry = _register_default_tools()
        names = registry.list()
        assert "list_files" in names
        assert "read_file" in names
        assert "write_file" in names
        assert "run_shell" in names
        assert len(names) == 4

    def test_web_search_registered_when_enabled(self):
        from miniclaw.agent.config import Config
        from miniclaw.cli import _register_default_tools

        config = Config()
        config.tools.allow_search = True
        registry = _register_default_tools(config)
        names = registry.list()
        assert "web_search" in names
        assert len(names) == 5

    def test_cli_allow_flags_override_registered_tools(self):
        from miniclaw.agent.config import Config
        from miniclaw.cli import _register_default_tools

        config = Config()
        args = Namespace(
            allow_file_write=True,
            allow_shell=True,
            allow_search=True,
        )
        registry = _register_default_tools(config, args)

        assert "web_search" in registry.list()
        assert registry.get("write_file").allow_write is True
        assert registry.get("run_shell").allow_shell is True


class TestDemoCommand:
    def test_context_compression(self, capsys):
        main(["demo", "context-compression"])
        captured = capsys.readouterr()
        assert "Before compression" in captured.out
        assert "After compression" in captured.out
        assert "token reduction" in captured.out

    def test_demo_parser(self):
        parser = build_parser()
        args = parser.parse_args(["demo", "context-compression"])
        assert args.command == "demo"
        assert args.demo_action == "context-compression"


class TestTraceExport:
    def _seed_db(self, db_path):
        """Seed a database with a session, messages, and traces."""
        from miniclaw.storage.sqlite_store import SQLiteStore
        import json

        with SQLiteStore(db_path) as store:
            sid = store.create_session("Test task")
            store.save_message(sid, "user", "Test task")
            store.save_trace(
                sid,
                step=1,
                event_json=json.dumps(
                    {
                        "parsed_action": "tool_call",
                        "tool_name": "list_files",
                        "arguments": {"path": "."},
                        "observation": {"entries": ["a.py", "b.py"]},
                    }
                ),
            )
            store.save_trace(
                sid,
                step=2,
                event_json=json.dumps(
                    {
                        "parsed_action": "final_answer",
                        "observation": "Found 2 files.",
                    }
                ),
            )
            return sid

    def test_export_stdout(self, tmp_path, capsys):
        db = tmp_path / "test.db"
        self._seed_db(db)
        main(["--db", str(db), "trace", "export"])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["session_id"] == 1
        assert data["task"] == "Test task"
        assert data["total_steps"] == 2
        assert data["steps"][0]["type"] == "tool_call"
        assert data["steps"][0]["tool_name"] == "list_files"
        assert data["steps"][1]["type"] == "final_answer"

    def test_export_to_file(self, tmp_path):
        import json

        db = tmp_path / "test.db"
        self._seed_db(db)
        out = tmp_path / "trace.json"
        main(["--db", str(db), "trace", "export", "-o", str(out)])
        assert out.exists()
        data = json.loads(out.read_text(encoding="utf-8"))
        assert data["total_steps"] == 2
        assert "steps" in data

    def test_export_with_session_id(self, tmp_path, capsys):
        import json

        db = tmp_path / "test.db"
        sid = self._seed_db(db)
        main(["--db", str(db), "trace", "export", "--session", str(sid)])
        captured = capsys.readouterr()
        data = json.loads(captured.out)
        assert data["session_id"] == sid

    def test_export_no_traces(self, tmp_path, capsys):
        from miniclaw.storage.sqlite_store import SQLiteStore

        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            store.create_session("Empty")
        main(["--db", str(db), "trace", "export"])
        captured = capsys.readouterr()
        assert "No traces" in captured.out

    def test_export_parser(self):
        parser = build_parser()
        args = parser.parse_args(["trace", "export", "--session", "5", "-o", "out.json"])
        assert args.command == "trace"
        assert args.trace_action == "export"
        assert args.session == 5
        assert args.output == "out.json"


class TestDoctorCommand:
    def test_doctor_prints_environment(self, tmp_path, capsys):
        db = tmp_path / "doctor.db"
        main(["--db", str(db), "doctor"])
        captured = capsys.readouterr()
        assert "MiniClaw doctor" in captured.out
        assert "Python:" in captured.out
        assert "LLM provider:" in captured.out
        assert "Database:" in captured.out
        assert "File writes:" in captured.out
        assert "Shell:" in captured.out
        assert "Web search:" in captured.out
