"""SQLite-backed long-term memory scoped by user and project."""

from __future__ import annotations

import hashlib
import re
from typing import Any

from miniclaw.memory.base import MemoryBackend
from miniclaw.storage.sqlite_store import SQLiteStore


class SQLiteMemoryBackend(MemoryBackend):
    """Persistent lightweight memory without a vector database dependency."""

    def __init__(self, store: SQLiteStore, project_id: str = "default") -> None:
        self._store = store
        self.project_id = project_id

    def add(self, text: str, user_id: str, metadata: dict[str, Any] | None = None) -> None:
        if not text.strip():
            return
        metadata = dict(metadata or {})
        key = str(metadata.pop("key", _memory_key(text)))
        source = str(metadata.pop("source", "agent"))
        self._store.upsert_agent_memory(
            user_id=user_id,
            project_id=self.project_id,
            memory_key=key,
            content=text,
            source=source,
            metadata=metadata,
        )

    def search(self, query: str, user_id: str, limit: int = 5) -> list[str]:
        rows = self._store.search_agent_memories(query, user_id, self.project_id, limit)
        if not rows:
            # SQLite has no tokenizer dependency here. Fall back to useful
            # query terms so a natural-language task can retrieve a memory
            # sharing only one keyword (including CJK characters).
            seen: dict[str, None] = {}
            for term in _search_terms(query):
                for row in self._store.search_agent_memories(term, user_id, self.project_id, limit):
                    seen.setdefault(row["content"], None)
                    if len(seen) >= limit:
                        break
                if len(seen) >= limit:
                    break
            return list(seen)
        return [row["content"] for row in rows]

    def remove(self, text: str, user_id: str) -> bool:
        return self._store.remove_agent_memory(user_id, self.project_id, text)


def _memory_key(text: str) -> str:
    """Provide a stable key so identical selective memories update in place."""
    normalized = " ".join(text.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:24]


def _search_terms(query: str) -> list[str]:
    """Return deterministic fallback terms without a full-text dependency."""
    terms = re.findall(r"[A-Za-z0-9_]{2,}|[\u4e00-\u9fff]", query.lower())
    return list(dict.fromkeys(terms))
