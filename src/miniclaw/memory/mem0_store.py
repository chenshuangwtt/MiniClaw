"""Mem0-backed memory backend.

Wraps the ``mem0ai`` package to provide long-term memory storage
with semantic search capabilities.

Falls back gracefully — if Mem0 is not installed or fails at runtime,
methods return safe defaults instead of crashing.

Usage::

    from miniclaw.memory.mem0_store import Mem0MemoryBackend

    backend = Mem0MemoryBackend()
    backend.add("用户喜欢简洁风格", user_id="alice")
    results = backend.search("风格偏好", user_id="alice")
"""

from __future__ import annotations

import logging
from typing import Any

from miniclaw.memory.base import MemoryBackend

logger = logging.getLogger(__name__)


class Mem0MemoryBackend(MemoryBackend):
    """Memory backend powered by Mem0.

    Args:
        config: Optional Mem0 config dict.  If ``None``, uses defaults.

    If mem0ai is not installed, the backend degrades to a no-op
    (``add`` does nothing, ``search`` returns ``[]``).
    """

    def __init__(self, config: dict[str, Any] | None = None) -> None:
        self._config = config
        self._memory = _init_mem0(config)

    @property
    def is_available(self) -> bool:
        """True if Mem0 was initialized successfully."""
        return self._memory is not None

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory via Mem0.

        If Mem0 is unavailable, silently does nothing.
        """
        if self._memory is None:
            return

        try:
            self._memory.add(text, user_id=user_id, metadata=metadata)
        except Exception as exc:
            logger.warning("Mem0 add failed: %s", exc)

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Semantic search via Mem0.

        Returns:
            List of matching memory texts, or ``[]`` on failure.
        """
        if self._memory is None:
            return []

        try:
            results = self._memory.search(query, user_id=user_id, limit=limit)
            return _extract_texts(results)
        except Exception as exc:
            logger.warning("Mem0 search failed: %s", exc)
            return []


def _init_mem0(config: dict[str, Any] | None):
    """Try to import and initialize Mem0.

    Returns the Memory instance or None if unavailable.
    """
    try:
        from mem0 import Memory
    except ImportError:
        logger.info("mem0ai not installed — Mem0MemoryBackend will be a no-op.")
        return None

    try:
        if config:
            return Memory.from_config(config)
        return Memory()
    except Exception as exc:
        logger.warning("Mem0 initialization failed: %s", exc)
        return None


def _extract_texts(results: Any) -> list[str]:
    """Normalize Mem0 search results into a flat list of strings.

    Mem0 returns results in various formats depending on version:
        - list of dicts: [{"memory": "text", "score": 0.9}, ...]
        - list of strings: ["text1", "text2", ...]
        - dict with "results" key: {"results": [...]}
        - other structures

    This function handles all cases gracefully.
    """
    if results is None:
        return []

    # If it's a dict with a "results" key, unwrap
    if isinstance(results, dict):
        results = results.get("results", results.get("memories", []))

    # If it's a single string, wrap
    if isinstance(results, str):
        return [results]

    # If it's a list, extract text from each item
    if not isinstance(results, list):
        return []

    texts: list[str] = []
    for item in results:
        if isinstance(item, str):
            texts.append(item)
        elif isinstance(item, dict):
            # Try common Mem0 response keys
            text = (
                item.get("memory")
                or item.get("text")
                or item.get("content")
                or item.get("value")
                or ""
            )
            if text:
                texts.append(str(text))
        else:
            texts.append(str(item))

    return texts
