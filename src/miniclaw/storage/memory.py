"""SQLite-backed persistent memory for conversations and tool calls."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any


class Memory:
    """Simple key-value + conversation history stored in SQLite.

    Tables:
        - kv: generic key-value store (session prefs, cached results).
        - messages: conversation history with role, content, metadata.

    Usage::

        mem = Memory(".miniclaw/miniclaw.db")
        mem.save("user_pref:language", "zh-CN")
        lang = mem.load("user_pref:language")

        mem.append_message("user", "What's the weather?")
        mem.append_message("assistant", "Let me check.")
        history = mem.get_messages(limit=10)
    """

    def __init__(self, db_path: str | Path = ":memory:") -> None:
        self._db_path = str(db_path)
        if self._db_path != ":memory:":
            Path(self._db_path).expanduser().parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self._db_path)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        cur = self._conn.cursor()
        cur.executescript("""
            CREATE TABLE IF NOT EXISTS kv (
                key   TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS messages (
                id         INTEGER PRIMARY KEY AUTOINCREMENT,
                role       TEXT NOT NULL,
                content    TEXT NOT NULL,
                metadata   TEXT DEFAULT '{}',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """)
        self._conn.commit()

    # ------------------------------------------------------------------
    # Key-Value
    # ------------------------------------------------------------------

    def save(self, key: str, value: Any) -> None:
        """Store a value (serialized as JSON) under *key*."""
        self._conn.execute(
            "INSERT OR REPLACE INTO kv (key, value) VALUES (?, ?)",
            (key, json.dumps(value, ensure_ascii=False)),
        )
        self._conn.commit()

    def load(self, key: str, default: Any = None) -> Any:
        """Load a value by *key*, returning *default* if not found."""
        row = self._conn.execute("SELECT value FROM kv WHERE key = ?", (key,)).fetchone()
        if row is None:
            return default
        return json.loads(row["value"])

    def delete(self, key: str) -> bool:
        """Delete a key. Returns True if it existed."""
        cur = self._conn.execute("DELETE FROM kv WHERE key = ?", (key,))
        self._conn.commit()
        return cur.rowcount > 0

    def keys(self, prefix: str = "") -> list[str]:
        """List all keys, optionally filtered by prefix."""
        if prefix:
            rows = self._conn.execute(
                "SELECT key FROM kv WHERE key LIKE ?", (f"{prefix}%",)
            ).fetchall()
        else:
            rows = self._conn.execute("SELECT key FROM kv").fetchall()
        return [row["key"] for row in rows]

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    def append_message(
        self, role: str, content: str, metadata: dict[str, Any] | None = None
    ) -> None:
        """Append a message to the conversation history."""
        self._conn.execute(
            "INSERT INTO messages (role, content, metadata) VALUES (?, ?, ?)",
            (role, content, json.dumps(metadata or {}, ensure_ascii=False)),
        )
        self._conn.commit()

    def get_messages(self, limit: int = 50) -> list[dict[str, Any]]:
        """Return the last *limit* messages in chronological order."""
        rows = self._conn.execute(
            "SELECT role, content, metadata, created_at FROM messages ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "role": row["role"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]

    def clear_messages(self) -> None:
        """Delete all conversation history."""
        self._conn.execute("DELETE FROM messages")
        self._conn.commit()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> Memory:
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
