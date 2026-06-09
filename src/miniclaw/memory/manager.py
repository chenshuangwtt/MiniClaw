"""MemoryManager — high-level coordinator for memory operations.

Sits between the AgentLoop and the memory backends.  Handles:
    - Conflict detection: deduplicate when a new memory overlaps an existing one.
    - Importance decay: reduce importance of memories that haven't been accessed.
    - Sensitive filtering: refuse to store API keys, passwords, etc.

Does NOT own a backend — receives one via ``__init__`` so the caller
controls which backend (Null, Mem0, Vector, SQLite) is used.

Usage::

    from miniclaw.memory.manager import MemoryManager
    from miniclaw.memory.mem0_store import Mem0MemoryBackend

    backend = Mem0MemoryBackend()
    manager = MemoryManager(backend)

    manager.add("用户偏好简洁风格", user_id="alice")
    results = manager.search("风格", user_id="alice")
    manager.decay(user_id="alice", decay_factor=0.95)
"""

from __future__ import annotations

import logging
from typing import Any

from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor, contains_sensitive

logger = logging.getLogger(__name__)


class MemoryManager:
    """High-level memory coordinator with conflict resolution and decay.

    Args:
        backend: The underlying storage backend.
        extractor: Memory extractor for deciding what to remember.
        similarity_threshold: Minimum score to consider a search result
            as conflicting with a new memory (backend-dependent).
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        extractor: MemoryExtractor | None = None,
        similarity_threshold: float = 0.8,
    ) -> None:
        self.backend = backend or NullMemoryBackend()
        self.extractor = extractor or MemoryExtractor()
        self.similarity_threshold = similarity_threshold

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
    ) -> bool:
        """Store a memory, with conflict detection and sensitive filtering.

        Args:
            text: The memory text.
            user_id: User identifier.
            metadata: Optional metadata.
            force: Skip conflict check if ``True``.

        Returns:
            ``True`` if the memory was stored, ``False`` if filtered out.
        """
        # 1. Sensitive filter
        if contains_sensitive(text):
            logger.info("Memory blocked: contains sensitive data.")
            return False

        # 2. Conflict detection
        if not force:
            existing = self._find_conflicts(text, user_id)
            if existing:
                logger.info("Memory skipped: conflicts with existing '%s'.", existing[:50])
                return False

        # 3. Store
        self.backend.add(text, user_id=user_id, metadata=metadata)
        return True

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Search for memories matching *query*."""
        return self.backend.search(query, user_id=user_id, limit=limit)

    def maybe_add_from_task(
        self,
        task: str,
        answer: str,
        user_id: str,
    ) -> bool:
        """Extract and store memories from a completed task.

        Uses the extractor to decide if the task is worth remembering,
        then calls ``add()`` with conflict detection.

        Returns:
            ``True`` if any memory was stored.
        """
        if not self.extractor.should_remember(task):
            return False

        fragments = self.extractor.extract(task)
        stored = False
        for fragment in fragments:
            if self.add(fragment, user_id=user_id):
                stored = True
        return stored

    def decay(
        self,
        user_id: str,
        decay_factor: float = 0.95,
        min_importance: int = 1,
    ) -> int:
        """Apply importance decay to all memories for *user_id*.

        For backends that support importance (e.g., SQLite), reduces
        each memory's importance by ``decay_factor``.  Memories at
        ``min_importance`` are left unchanged.

        Args:
            user_id: User identifier.
            decay_factor: Multiplier applied to importance (0.0–1.0).
            min_importance: Floor value — importance won't drop below this.

        Returns:
            Number of memories that were decayed.
        """
        # Only works with backends that expose a `decay` method
        if hasattr(self.backend, "decay"):
            return self.backend.decay(user_id, decay_factor, min_importance)

        logger.debug("Backend does not support decay — skipping.")
        return 0

    def _find_conflicts(self, text: str, user_id: str) -> str | None:
        """Check if a similar memory already exists.

        Returns the conflicting memory text, or ``None``.
        """
        try:
            existing = self.backend.search(text, user_id=user_id, limit=3)
            for mem in existing:
                if _text_overlap(text, mem) > self.similarity_threshold:
                    return mem
        except Exception:
            pass
        return None


def _text_overlap(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)
