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

import json
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
        # The engine performs backend operations through ``asyncio.to_thread``.
        # SQLite remains process-local, but this permits that one connection to
        # serve the sequential worker calls used by the lightweight runtime.
        self._conn = sqlite3.connect(self._db_path, check_same_thread=False)
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
                key         TEXT NOT NULL,
                value       TEXT NOT NULL DEFAULT '',
                user_id     TEXT NOT NULL DEFAULT 'default',
                importance  INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, key)
            );

            CREATE TABLE IF NOT EXISTS traces (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  INTEGER NOT NULL,
                step        INTEGER NOT NULL,
                event_json  TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS runs (
                id          TEXT PRIMARY KEY,
                session_id  INTEGER,
                task        TEXT NOT NULL,
                status      TEXT NOT NULL,
                error       TEXT,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at  TEXT NOT NULL DEFAULT (datetime('now')),
                FOREIGN KEY (session_id) REFERENCES sessions(id)
            );

            CREATE TABLE IF NOT EXISTS checkpoints (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id      TEXT NOT NULL,
                sequence    INTEGER NOT NULL,
                state_json  TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(run_id, sequence),
                FOREIGN KEY (run_id) REFERENCES runs(id)
            );

            CREATE TABLE IF NOT EXISTS agent_memories (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id       TEXT NOT NULL,
                project_id    TEXT NOT NULL,
                memory_key    TEXT NOT NULL,
                content       TEXT NOT NULL,
                source        TEXT NOT NULL DEFAULT 'agent',
                metadata_json TEXT NOT NULL DEFAULT '{}',
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at    TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(user_id, project_id, memory_key)
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
    # Durable agent runs and checkpoints
    # ------------------------------------------------------------------

    def create_run(self, run_id: str, task: str, session_id: int | None = None) -> None:
        """Create a durable agent run in the ``created`` state."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO runs (id, session_id, task, status) VALUES (?, ?, ?, 'created')",
            (run_id, session_id, task),
        )
        conn.commit()

    def update_run(self, run_id: str, status: str, error: str | None = None) -> None:
        """Persist a lifecycle transition for an existing run."""
        conn = self._get_conn()
        conn.execute(
            "UPDATE runs SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (status, error, _now(), run_id),
        )
        conn.commit()

    def get_run(self, run_id: str) -> dict[str, Any] | None:
        """Return one durable run, or ``None`` when it does not exist."""
        row = (
            self._get_conn()
            .execute(
                "SELECT id, session_id, task, status, error, created_at, updated_at "
                "FROM runs WHERE id = ?",
                (run_id,),
            )
            .fetchone()
        )
        return dict(row) if row else None

    def list_runs(self, session_id: int | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """List recent durable runs, optionally scoped to a session."""
        conn = self._get_conn()
        if session_id is None:
            rows = conn.execute(
                "SELECT id, session_id, task, status, error, created_at, updated_at "
                "FROM runs ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, session_id, task, status, error, created_at, updated_at "
                "FROM runs WHERE session_id = ? ORDER BY created_at DESC LIMIT ?",
                (session_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def save_checkpoint(self, run_id: str, sequence: int, state: dict[str, Any]) -> None:
        """Atomically store a JSON-serializable recovery checkpoint."""
        conn = self._get_conn()
        conn.execute(
            "INSERT INTO checkpoints (run_id, sequence, state_json) VALUES (?, ?, ?) "
            "ON CONFLICT(run_id, sequence) DO UPDATE SET state_json = excluded.state_json",
            (run_id, sequence, json.dumps(state, ensure_ascii=False, default=str)),
        )
        conn.commit()

    def get_latest_checkpoint(self, run_id: str) -> dict[str, Any] | None:
        """Return the latest checkpoint decoded from JSON."""
        row = (
            self._get_conn()
            .execute(
                "SELECT sequence, state_json, created_at FROM checkpoints "
                "WHERE run_id = ? ORDER BY sequence DESC LIMIT 1",
                (run_id,),
            )
            .fetchone()
        )
        if row is None:
            return None
        return {
            "sequence": row["sequence"],
            "state": json.loads(row["state_json"]),
            "created_at": row["created_at"],
        }

    # ------------------------------------------------------------------
    # Scoped long-term agent memories
    # ------------------------------------------------------------------

    def upsert_agent_memory(
        self,
        user_id: str,
        project_id: str,
        memory_key: str,
        content: str,
        source: str = "agent",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Create or update a memory isolated by user and project."""
        conn = self._get_conn()
        now = _now()
        conn.execute(
            "INSERT INTO agent_memories "
            "(user_id, project_id, memory_key, content, source, metadata_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, project_id, memory_key) DO UPDATE SET "
            "content = excluded.content, source = excluded.source, "
            "metadata_json = excluded.metadata_json, updated_at = excluded.updated_at",
            (
                user_id,
                project_id,
                memory_key,
                content,
                source,
                json.dumps(metadata or {}, ensure_ascii=False),
                now,
                now,
            ),
        )
        conn.commit()

    def search_agent_memories(
        self, query: str, user_id: str, project_id: str, limit: int = 5
    ) -> list[dict[str, Any]]:
        """Search scoped memories using a deterministic SQLite substring match."""
        pattern = f"%{query}%"
        rows = (
            self._get_conn()
            .execute(
                "SELECT id, memory_key, content, source, metadata_json, created_at, updated_at "
                "FROM agent_memories WHERE user_id = ? AND project_id = ? "
                "AND (content LIKE ? OR memory_key LIKE ?) "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (user_id, project_id, pattern, pattern, limit),
            )
            .fetchall()
        )
        return [
            {
                **dict(row),
                "metadata": json.loads(row["metadata_json"]),
            }
            for row in rows
        ]

    def remove_agent_memory(self, user_id: str, project_id: str, content: str) -> bool:
        """Remove an exact scoped memory and report whether it existed."""
        cur = self._get_conn().execute(
            "DELETE FROM agent_memories WHERE user_id = ? AND project_id = ? AND content = ?",
            (user_id, project_id, content),
        )
        self._get_conn().commit()
        return cur.rowcount > 0

    def list_agent_memories(
        self, user_id: str, project_id: str, limit: int = 50
    ) -> list[dict[str, Any]]:
        """List scoped agent memories newest first for local inspection."""
        rows = (
            self._get_conn()
            .execute(
                "SELECT id, memory_key, content, source, metadata_json, created_at, updated_at "
                "FROM agent_memories WHERE user_id = ? AND project_id = ? "
                "ORDER BY updated_at DESC, id DESC LIMIT ?",
                (user_id, project_id, limit),
            )
            .fetchall()
        )
        return [{**dict(row), "metadata": json.loads(row["metadata_json"])} for row in rows]

    # ------------------------------------------------------------------
    # Memories
    # ------------------------------------------------------------------

    def save_memory(
        self, key: str, value: str, importance: int = 1, user_id: str = "default"
    ) -> int:
        """Save or update a memory entry.

        If *key* already exists, the value and importance are updated.

        Args:
            key: Unique key (e.g., ``"user:name"``).
            value: The value to store.
            importance: 1–10 scale for retrieval ranking.
            user_id: User identifier.

        Returns:
            The memory's ``id``.
        """
        conn = self._get_conn()
        now = _now()
        conn.execute(
            "INSERT INTO memories (key, value, user_id, importance, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(user_id, key) DO UPDATE SET "
            "  value = excluded.value, "
            "  importance = excluded.importance, "
            "  updated_at = excluded.updated_at",
            (key, value, user_id, importance, now, now),
        )
        conn.commit()
        # Fetch the id (either newly inserted or existing)
        row = conn.execute(
            "SELECT id FROM memories WHERE user_id = ? AND key = ?",
            (user_id, key),
        ).fetchone()
        return row["id"]  # type: ignore[index]

    def list_memories(self, limit: int = 50, user_id: str | None = None) -> list[dict[str, Any]]:
        """Return all memories ordered by importance (desc), then by key.

        Args:
            limit: Maximum number of entries.
            user_id: If provided, filter to this user's memories.

        Returns:
            List of memory dicts.
        """
        conn = self._get_conn()
        if user_id is not None:
            rows = conn.execute(
                "SELECT id, key, value, user_id, importance, created_at, updated_at "
                "FROM memories WHERE user_id = ? "
                "ORDER BY importance DESC, key ASC LIMIT ?",
                (user_id, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, key, value, user_id, importance, created_at, updated_at "
                "FROM memories ORDER BY importance DESC, key ASC LIMIT ?",
                (limit,),
            ).fetchall()
        return [dict(row) for row in rows]

    def search_memories(
        self,
        query: str,
        limit: int = 10,
        user_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Simple keyword search across memory keys and values.

        Uses ``LIKE %query%`` for substring matching (case-insensitive
        in SQLite's default configuration).

        Args:
            query: Search term.
            limit: Maximum results.
            user_id: If provided, filter to this user's memories.

        Returns:
            List of matching memory dicts.
        """
        conn = self._get_conn()
        pattern = f"%{query}%"
        if user_id is not None:
            rows = conn.execute(
                "SELECT id, key, value, user_id, importance, created_at, updated_at "
                "FROM memories "
                "WHERE user_id = ? AND (key LIKE ? OR value LIKE ?) "
                "ORDER BY importance DESC, key ASC LIMIT ?",
                (user_id, pattern, pattern, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT id, key, value, user_id, importance, created_at, updated_at "
                "FROM memories "
                "WHERE key LIKE ? OR value LIKE ? "
                "ORDER BY importance DESC, key ASC LIMIT ?",
                (pattern, pattern, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def decay(
        self,
        user_id: str = "default",
        decay_factor: float = 0.95,
        min_importance: int = 1,
    ) -> int:
        """Apply importance decay to all memories for *user_id*.

        Reduces each memory's importance by ``decay_factor``.  Memories at
        or below ``min_importance`` are left unchanged.

        Args:
            user_id: User identifier.
            decay_factor: Multiplier applied to importance (0.0–1.0).
            min_importance: Floor value — importance won't drop below this.

        Returns:
            Number of memories that were decayed.
        """
        conn = self._get_conn()
        now = _now()
        # Count affected rows first
        count_row = conn.execute(
            "SELECT COUNT(*) as cnt FROM memories WHERE user_id = ? AND importance > ?",
            (user_id, min_importance),
        ).fetchone()
        count = count_row["cnt"]  # type: ignore[index]

        conn.execute(
            "UPDATE memories "
            "SET importance = MAX(?, CAST(importance * ? AS INTEGER)), "
            "    updated_at = ? "
            "WHERE user_id = ? AND importance > ?",
            (min_importance, decay_factor, now, user_id, min_importance),
        )
        conn.commit()
        return count

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
