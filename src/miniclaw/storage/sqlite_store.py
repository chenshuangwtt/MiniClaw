"""SQLite-backed persistent storage for sessions, messages, memories, and traces.

Uses Python's built-in ``sqlite3`` module — no external dependencies.
All writes use parameterized SQL to prevent injection.

Tables::

    sessions  — conversation sessions
    messages  — per-session message history
    memories  — key-value long-term memory with importance
    traces    — per-session step-by-step event log

Usage::

    with SQLiteStore(".miniclaw/miniclaw.db") as store:
        sid = store.create_session("Weather task")
        store.save_message(sid, "user", "What's the weather?")
        store.save_message(sid, "assistant", "Let me check.")
        msgs = store.list_messages(sid)

        store.save_memory("user:name", "Alice", importance=5)
        results = store.search_memories("alice")

        store.save_trace(sid, step=1, event_json='{"action": "tool_call"}')
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class SQLiteStore:
    """Lightweight SQLite storage for MiniClaw's persistent data.

    Args:
        db_path: Path to the SQLite database file.  Use ``":memory:"``
            for an in-memory database (useful for testing).
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def connect(self) -> None:
        """Open the database connection and initialize tables."""
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")

    def init_db(self) -> None:
        """Create all tables if they don't already exist.

        Safe to call multiple times — uses ``CREATE TABLE IF NOT EXISTS``.
        """
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS sessions (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                title       TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS messages (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                role        TEXT NOT NULL,
                content     TEXT NOT NULL DEFAULT '',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS memories (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                key         TEXT NOT NULL UNIQUE,
                value       TEXT NOT NULL DEFAULT '',
                importance  INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS traces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                step        INTEGER NOT NULL,
                event_json  TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );
        """)
        conn.commit()

    def close(self) -> None:
        """Close the database connection."""
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SQLiteStore:
        self.connect()
        self.init_db()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------
    # Sessions
    # ------------------------------------------------------------------

    def create_session(self, title: str = "") -> int:
        """Create a new session and return its ID.

        Args:
            title: Optional human-readable title.

        Returns:
            The new session's ``id``.
        """
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO sessions (title) VALUES (?)",
            (title,),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def save_message(self, session_id: int, role: str, content: str) -> int:
        """Append a message to a session.

        Args:
            session_id: The session to attach this message to.
            role: Message role (``system``, ``user``, ``assistant``, ``tool``).
            content: Message text.

        Returns:
            The new message's ``id``.
        """
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
            (session_id, role, content),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_messages(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return messages for a session in chronological order.

        Args:
            session_id: The session to query.
            limit: Maximum number of messages to return.

        Returns:
            List of message dicts with ``id``, ``role``, ``content``, ``created_at``.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, role, content, created_at "
            "FROM messages WHERE session_id = ? "
            "ORDER BY id ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def save_memory(self, key: str, value: str, importance: int = 1) -> int:
        """Save or update a memory entry.

        If *key* already exists, the value and importance are updated.

        Args:
            key: Unique key (e.g., ``"user:name"``).
            value: The value to store.
            importance: 1–10 scale for retrieval ranking.

        Returns:
            The memory's ``id``.
        """
        conn = self._get_conn()
        now = _now()
        conn.execute(
            "INSERT INTO memories (key, value, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(key) DO UPDATE SET "
            "  value = excluded.value, "
            "  importance = excluded.importance, "
            "  updated_at = excluded.updated_at",
            (key, value, importance, now, now),
        )
        conn.commit()
        # Fetch the id (either newly inserted or existing)
        row = conn.execute("SELECT id FROM memories WHERE key = ?", (key,)).fetchone()
        return row["id"]  # type: ignore[index]

    def list_memories(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return all memories ordered by importance (desc), then by key.

        Args:
            limit: Maximum number of entries.

        Returns:
            List of memory dicts.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, key, value, importance, created_at, updated_at "
            "FROM memories ORDER BY importance DESC, key ASC LIMIT ?",
            (limit,),
        ).fetchall()
        return [dict(row) for row in rows]

    def search_memories(self, query: str, limit: int = 10) -> list[dict[str, Any]]:
        """Simple keyword search across memory keys and values.

        Uses ``LIKE %query%`` for substring matching (case-insensitive
        in SQLite's default configuration).

        Args:
            query: Search term.
            limit: Maximum results.

        Returns:
            List of matching memory dicts.
        """
        conn = self._get_conn()
        pattern = f"%{query}%"
        rows = conn.execute(
            "SELECT id, key, value, importance, created_at, updated_at "
            "FROM memories "
            "WHERE key LIKE ? OR value LIKE ? "
            "ORDER BY importance DESC, key ASC LIMIT ?",
            (pattern, pattern, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Traces
    # ------------------------------------------------------------------

    def save_trace(self, session_id: int, step: int, event_json: str) -> int:
        """Save a trace event for a session step.

        Args:
            session_id: The session this trace belongs to.
            step: Step number (1-indexed).
            event_json: JSON string describing the event.

        Returns:
            The new trace's ``id``.
        """
        conn = self._get_conn()
        cur = conn.execute(
            "INSERT INTO traces (session_id, step, event_json) VALUES (?, ?, ?)",
            (session_id, step, event_json),
        )
        conn.commit()
        return cur.lastrowid  # type: ignore[return-value]

    def list_traces(self, session_id: int, limit: int = 100) -> list[dict[str, Any]]:
        """Return traces for a session in step order.

        Args:
            session_id: The session to query.
            limit: Maximum number of traces.

        Returns:
            List of trace dicts with ``id``, ``step``, ``event_json``, ``created_at``.
        """
        conn = self._get_conn()
        rows = conn.execute(
            "SELECT id, step, event_json, created_at "
            "FROM traces WHERE session_id = ? "
            "ORDER BY step ASC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [dict(row) for row in rows]

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _get_conn(self) -> sqlite3.Connection:
        """Return the active connection, auto-connecting if needed."""
        if self._conn is None:
            self.connect()
            self.init_db()
        assert self._conn is not None
        return self._conn


def _now() -> str:
    """UTC timestamp in ISO-like format."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
