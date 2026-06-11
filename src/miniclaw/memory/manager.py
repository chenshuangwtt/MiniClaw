"""MemoryManager — high-level coordinator for memory operations.

Sits between the AgentLoop and the memory backends.  Handles:
    - Conflict detection: deduplicate when a new memory overlaps an existing one.
    - Conflict resolution strategies: skip, replace, or merge.
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

# Conflict resolution strategies
CONFLICT_SKIP = "skip"
CONFLICT_REPLACE = "replace"
CONFLICT_MERGE = "merge"


class MemoryManager:
    """High-level memory coordinator with conflict resolution and decay.

    Args:
        backend: The underlying storage backend.
        extractor: Memory extractor for deciding what to remember.
        similarity_threshold: Minimum Jaccard score to consider a search result
            as conflicting with a new memory.
        default_conflict_strategy: Default strategy when a conflict is detected.
            One of ``"skip"`` (default), ``"replace"``, or ``"merge"``.
    """

    def __init__(
        self,
        backend: MemoryBackend | None = None,
        extractor: MemoryExtractor | None = None,
        similarity_threshold: float = 0.8,
        default_conflict_strategy: str = CONFLICT_SKIP,
    ) -> None:
        self.backend = backend or NullMemoryBackend()
        self.extractor = extractor or MemoryExtractor()
        self.similarity_threshold = similarity_threshold
        self.default_conflict_strategy = default_conflict_strategy

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
        force: bool = False,
        conflict_strategy: str | None = None,
    ) -> bool:
        """Store a memory, with conflict detection and sensitive filtering.

        Args:
            text: The memory text.
            user_id: User identifier.
            metadata: Optional metadata.
            force: Skip conflict check if ``True``.
            conflict_strategy: Override the default conflict strategy for this
                call.  One of ``"skip"``, ``"replace"``, ``"merge"``.

        Returns:
            ``True`` if the memory was stored, ``False`` if filtered out.
        """
        # 1. Sensitive filter
        if contains_sensitive(text):
            logger.info("Memory blocked: contains sensitive data.")
            return False

        # 2. Conflict detection
        strategy = conflict_strategy or self.default_conflict_strategy

        if not force:
            conflict = self._find_conflict(text, user_id)
            if conflict is not None:
                return self._handle_conflict(text, conflict, user_id, metadata, strategy)

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

    def _find_conflict(self, text: str, user_id: str) -> str | None:
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

    def _handle_conflict(
        self,
        new_text: str,
        existing_text: str,
        user_id: str,
        metadata: dict[str, Any] | None,
        strategy: str,
    ) -> bool:
        """Handle a conflict based on the chosen strategy.

        Returns ``True`` if the new memory was ultimately stored.
        """
        if strategy == CONFLICT_REPLACE:
            # Remove the old memory and store the new one
            try:
                self.backend.remove(existing_text, user_id)
            except Exception as exc:
                logger.warning("Failed to remove conflicting memory: %s", exc)
            self.backend.add(new_text, user_id=user_id, metadata=metadata)
            logger.info("Memory replaced: '%s' → '%s'.", existing_text[:40], new_text[:40])
            return True

        if strategy == CONFLICT_MERGE:
            # Combine the two memories into one
            merged = _merge_texts(existing_text, new_text)
            try:
                self.backend.remove(existing_text, user_id)
            except Exception as exc:
                logger.warning("Failed to remove old memory for merge: %s", exc)
            self.backend.add(merged, user_id=user_id, metadata=metadata)
            logger.info("Memory merged: '%s' + '%s'.", existing_text[:30], new_text[:30])
            return True

        # Default: skip
        logger.info("Memory skipped: conflicts with existing '%s'.", existing_text[:50])
        return False


def _text_overlap(a: str, b: str) -> float:
    """Simple word-level Jaccard similarity between two strings."""
    words_a = set(a.lower().split())
    words_b = set(b.lower().split())
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


def _merge_texts(existing: str, new: str) -> str:
    """Merge two memory texts into one.

    Simple strategy: concatenate with a separator if they are different enough,
    or take the longer one if they are nearly identical.
    """
    overlap = _text_overlap(existing, new)
    if overlap > 0.9:
        # Nearly identical — take the longer (presumably more detailed) one
        return existing if len(existing) >= len(new) else new
    # Different enough — concatenate
    return f"{existing}; {new}"
