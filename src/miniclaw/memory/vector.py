"""Lightweight vector-backed memory backend.

The default embedder is deliberately simple and dependency-free: it builds a
hashed bag-of-words vector and ranks memories by cosine similarity. Production
users can pass a real embedding function without changing the backend API.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from miniclaw.memory.base import MemoryBackend


EmbeddingFn = Callable[[str], list[float]]


@dataclass
class VectorMemoryEntry:
    """A stored text memory and its vector representation."""

    text: str
    user_id: str
    vector: list[float]
    metadata: dict[str, Any] = field(default_factory=dict)


class VectorMemoryBackend(MemoryBackend):
    """In-memory semantic-ish retrieval backend.

    This backend is useful for demos and tests where Mem0 or an external vector
    database would be too heavy. It is not persistent by design.
    """

    def __init__(
        self,
        embedder: EmbeddingFn | None = None,
        dimensions: int = 128,
    ) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be greater than zero")
        self.dimensions = dimensions
        self._embedder = embedder or (lambda text: hash_embed(text, dimensions))
        self._entries: list[VectorMemoryEntry] = []

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory entry with its embedding."""
        if not text.strip():
            return
        vector = self._embedder(text)
        self._entries.append(
            VectorMemoryEntry(
                text=text,
                user_id=user_id,
                vector=vector,
                metadata=dict(metadata or {}),
            )
        )

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Return memories ranked by cosine similarity to *query*."""
        if limit <= 0 or not query.strip():
            return []

        query_vector = self._embedder(query)
        scored: list[tuple[float, int, str]] = []
        for index, entry in enumerate(self._entries):
            if entry.user_id != user_id:
                continue
            score = cosine_similarity(query_vector, entry.vector)
            if score > 0:
                scored.append((score, index, entry.text))

        scored.sort(key=lambda item: (-item[0], item[1]))
        return [text for _, _, text in scored[:limit]]

    def entries(self, user_id: str | None = None) -> list[VectorMemoryEntry]:
        """Return a copy of stored entries, optionally filtered by user."""
        if user_id is None:
            return list(self._entries)
        return [entry for entry in self._entries if entry.user_id == user_id]


def hash_embed(text: str, dimensions: int = 128) -> list[float]:
    """Embed text as a normalized hashed bag-of-words vector."""
    if dimensions <= 0:
        raise ValueError("dimensions must be greater than zero")

    vector = [0.0] * dimensions
    for token in _tokenize(text):
        vector[hash(token) % dimensions] += 1.0

    norm = math.sqrt(sum(value * value for value in vector))
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def cosine_similarity(left: list[float], right: list[float]) -> float:
    """Return cosine similarity for two vectors."""
    if not left or not right or len(left) != len(right):
        return 0.0

    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        return 0.0

    dot = sum(a * b for a, b in zip(left, right, strict=True))
    return dot / (left_norm * right_norm)


def _tokenize(text: str) -> list[str]:
    """Tokenize English words/numbers and individual CJK characters."""
    return re.findall(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]", text.lower())
