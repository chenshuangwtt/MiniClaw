"""Tests for enhanced conflict resolution in MemoryManager."""

from __future__ import annotations

from typing import Any


from miniclaw.memory.base import MemoryBackend
from miniclaw.memory.composite import CompositeMemoryBackend
from miniclaw.memory.manager import (
    CONFLICT_MERGE,
    CONFLICT_REPLACE,
    MemoryManager,
    _merge_texts,
)
from miniclaw.memory.vector import VectorMemoryBackend


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class SimpleBackend(MemoryBackend):
    """In-memory backend for testing conflict resolution."""

    def __init__(self) -> None:
        self._store: dict[str, list[str]] = {}  # user_id -> [texts]

    def add(self, text: str, user_id: str, metadata: dict[str, Any] | None = None) -> None:
        self._store.setdefault(user_id, []).append(text)

    def search(self, query: str, user_id: str, limit: int = 5) -> list[str]:
        return self._store.get(user_id, [])[:limit]

    def remove(self, text: str, user_id: str) -> bool:
        texts = self._store.get(user_id, [])
        if text in texts:
            texts.remove(text)
            return True
        return False


# ---------------------------------------------------------------------------
# _merge_texts
# ---------------------------------------------------------------------------


class TestMergeTexts:
    def test_nearly_identical_takes_longer(self) -> None:
        # Jaccard("aaa bbb", "aaa bbb ccc") = 2/3 ≈ 0.67 → concat
        result = _merge_texts("aaa bbb", "aaa bbb ccc")
        assert result == "aaa bbb; aaa bbb ccc"

    def test_very_similar_takes_longer(self) -> None:
        # Jaccard = 1.0 (identical word sets) → takes longer
        result = _merge_texts("same words here", "same words here")
        assert result == "same words here"

    def test_different_concats(self) -> None:
        result = _merge_texts("I like cats", "I prefer metric units")
        assert result == "I like cats; I prefer metric units"

    def test_identical_returns_one(self) -> None:
        result = _merge_texts("same text", "same text")
        assert result == "same text"


# ---------------------------------------------------------------------------
# Conflict strategies
# ---------------------------------------------------------------------------


class TestConflictSkip:
    """Default strategy: skip the new memory on conflict."""

    def test_skip_on_conflict(self) -> None:
        backend = SimpleBackend()
        manager = MemoryManager(backend, similarity_threshold=0.5)
        backend.add("I like dark mode", user_id="alice")

        result = manager.add("I like dark mode settings", user_id="alice")
        # Should be skipped due to high overlap
        assert result is False

    def test_no_conflict_stores(self) -> None:
        backend = SimpleBackend()
        manager = MemoryManager(backend, similarity_threshold=0.8)

        result = manager.add("I like dark mode", user_id="alice")
        assert result is True
        assert "I like dark mode" in backend._store["alice"]


class TestConflictReplace:
    """Replace strategy: remove old, store new."""

    def test_replace_on_conflict(self) -> None:
        backend = SimpleBackend()
        manager = MemoryManager(
            backend,
            similarity_threshold=0.5,
            default_conflict_strategy=CONFLICT_REPLACE,
        )
        backend.add("I like dark mode", user_id="alice")

        result = manager.add("I like dark mode a lot", user_id="alice")
        assert result is True
        texts = backend._store["alice"]
        assert "I like dark mode" not in texts
        assert "I like dark mode a lot" in texts

    def test_replace_no_conflict_just_adds(self) -> None:
        backend = SimpleBackend()
        manager = MemoryManager(
            backend,
            similarity_threshold=0.8,
            default_conflict_strategy=CONFLICT_REPLACE,
        )

        result = manager.add("brand new fact", user_id="alice")
        assert result is True
        assert "brand new fact" in backend._store["alice"]


class TestConflictMerge:
    """Merge strategy: combine old and new into one memory."""

    def test_merge_on_conflict(self) -> None:
        # "I like cats" vs "I prefer metric units": Jaccard ≈ 0.14
        backend = SimpleBackend()
        manager = MemoryManager(
            backend,
            similarity_threshold=0.1,  # low threshold to trigger merge
            default_conflict_strategy=CONFLICT_MERGE,
        )
        backend.add("I like cats", user_id="alice")

        result = manager.add("I prefer metric units", user_id="alice")
        assert result is True
        texts = backend._store["alice"]
        # Should have the merged text, not the original two
        assert len(texts) == 1
        assert "cats" in texts[0]
        assert "metric" in texts[0]

    def test_merge_nearly_identical_takes_longer(self) -> None:
        # "dark mode" vs "dark mode preference enabled": Jaccard ≈ 0.33
        backend = SimpleBackend()
        manager = MemoryManager(
            backend,
            similarity_threshold=0.2,
            default_conflict_strategy=CONFLICT_MERGE,
        )
        backend.add("dark mode", user_id="alice")

        result = manager.add("dark mode preference enabled", user_id="alice")
        assert result is True
        texts = backend._store["alice"]
        assert len(texts) == 1
        assert "preference enabled" in texts[0]


class TestPerCallStrategy:
    """Conflict strategy can be overridden per add() call."""

    def test_override_to_replace(self) -> None:
        backend = SimpleBackend()
        manager = MemoryManager(backend, similarity_threshold=0.5)  # default: skip
        backend.add("I like dark mode", user_id="alice")

        result = manager.add(
            "I like dark mode a lot",
            user_id="alice",
            conflict_strategy=CONFLICT_REPLACE,
        )
        assert result is True
        texts = backend._store["alice"]
        assert "I like dark mode a lot" in texts

    def test_override_to_merge(self) -> None:
        # "fact A" vs "completely different fact B": Jaccard ≈ 0.2
        backend = SimpleBackend()
        manager = MemoryManager(backend, similarity_threshold=0.1)  # default: skip
        backend.add("fact A", user_id="alice")

        result = manager.add(
            "completely different fact B",
            user_id="alice",
            conflict_strategy=CONFLICT_MERGE,
        )
        assert result is True


# ---------------------------------------------------------------------------
# VectorMemoryBackend integration
# ---------------------------------------------------------------------------


class TestVectorConflictIntegration:
    """Test conflict resolution with VectorMemoryBackend (has remove)."""

    def test_replace_with_vector(self) -> None:
        backend = VectorMemoryBackend()
        manager = MemoryManager(
            backend,
            similarity_threshold=0.3,
            default_conflict_strategy=CONFLICT_REPLACE,
        )
        manager.add("I like dark mode", user_id="alice")
        manager.add("I like dark mode enabled", user_id="alice")

        entries = backend.entries(user_id="alice")
        texts = [e.text for e in entries]
        # The first should have been replaced
        assert "I like dark mode" not in texts
        assert "I like dark mode enabled" in texts


class TestCompositeConflictIntegration:
    """Test conflict resolution with CompositeMemoryBackend."""

    def test_replace_removes_old_memory_from_both_backends(self) -> None:
        primary = VectorMemoryBackend()
        secondary = VectorMemoryBackend()
        backend = CompositeMemoryBackend(primary, secondary)
        manager = MemoryManager(
            backend,
            similarity_threshold=0.3,
            default_conflict_strategy=CONFLICT_REPLACE,
        )

        manager.add("I like dark mode", user_id="alice")
        manager.add("I like dark mode enabled", user_id="alice")

        for child in (primary, secondary):
            texts = [e.text for e in child.entries(user_id="alice")]
            assert "I like dark mode" not in texts
            assert "I like dark mode enabled" in texts
