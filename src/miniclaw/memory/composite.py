"""Composite memory backend — writes to multiple backends, merges search results.

Useful for combining a fast local backend (e.g., VectorMemoryBackend) with a
semantic backend (e.g., Mem0MemoryBackend) to get the best of both worlds.

Usage::

    from miniclaw.memory.composite import CompositeMemoryBackend
    from miniclaw.memory.mem0_store import Mem0MemoryBackend
    from miniclaw.memory.vector import VectorMemoryBackend

    backend = CompositeMemoryBackend(
        primary=Mem0MemoryBackend(),
        secondary=VectorMemoryBackend(),
    )
    backend.add("用户喜欢简洁风格", user_id="alice")
    results = backend.search("风格偏好", user_id="alice")
"""

from __future__ import annotations

import logging
from typing import Any

from miniclaw.memory.base import MemoryBackend

logger = logging.getLogger(__name__)


def _text_jaccard(a: str, b: str) -> float:
    """Word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


class CompositeMemoryBackend(MemoryBackend):
    """Memory backend that delegates to two underlying backends.

    - ``add()`` writes to **both** backends.
    - ``search()`` merges results from both, deduplicates by Jaccard
      similarity, and returns up to ``limit`` items.

    Args:
        primary: The primary backend (typically the semantic one).
        secondary: The secondary backend (typically the fast/local one).
        dedup_threshold: Jaccard similarity above which two results are
            considered duplicates (default 0.7).
    """

    def __init__(
        self,
        primary: MemoryBackend,
        secondary: MemoryBackend,
        dedup_threshold: float = 0.7,
    ) -> None:
        self.primary = primary
        self.secondary = secondary
        self.dedup_threshold = dedup_threshold

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory in both backends.

        If one backend fails, the other is still written to.
        """
        for name, backend in [("primary", self.primary), ("secondary", self.secondary)]:
            try:
                backend.add(text, user_id=user_id, metadata=metadata)
            except Exception as exc:
                logger.warning("CompositeMemoryBackend.%s add failed: %s", name, exc)

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Search both backends and merge results.

        Deduplicates by Jaccard similarity.  Primary results appear first.
        """
        seen: list[str] = []

        for name, backend in [("primary", self.primary), ("secondary", self.secondary)]:
            try:
                results = backend.search(query, user_id=user_id, limit=limit)
            except Exception as exc:
                logger.warning("CompositeMemoryBackend.%s search failed: %s", name, exc)
                continue

            for text in results:
                if not self._is_duplicate(text, seen):
                    seen.append(text)

        return seen[:limit]

    def remove(self, text: str, user_id: str) -> bool:
        """Remove a memory from both backends.

        Returns ``True`` if at least one backend removed an entry.  Backends
        that do not support removal are treated as a non-fatal no-op.
        """
        removed = False
        for name, backend in [("primary", self.primary), ("secondary", self.secondary)]:
            try:
                removed = backend.remove(text, user_id=user_id) or removed
            except Exception as exc:
                logger.warning("CompositeMemoryBackend.%s remove failed: %s", name, exc)
        return removed

    def _is_duplicate(self, text: str, existing: list[str]) -> bool:
        """Check if *text* is similar enough to any entry in *existing*."""
        for entry in existing:
            if _text_jaccard(text, entry) >= self.dedup_threshold:
                return True
        return False
