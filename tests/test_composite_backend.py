"""Tests for CompositeMemoryBackend."""

from __future__ import annotations

from typing import Any


from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.composite import CompositeMemoryBackend, _text_jaccard


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class RecordingBackend(MemoryBackend):
    """Backend that records calls for assertion."""

    def __init__(self) -> None:
        self.added: list[tuple[str, str]] = []
        self.search_results: list[str] = []

    def add(self, text: str, user_id: str, metadata: dict[str, Any] | None = None) -> None:
        self.added.append((text, user_id))

    def search(self, query: str, user_id: str, limit: int = 5) -> list[str]:
        return self.search_results[:limit]

    def remove(self, text: str, user_id: str) -> bool:
        if text in self.search_results:
            self.search_results.remove(text)
            return True
        return False


class FailingBackend(MemoryBackend):
    """Backend that always raises."""

    def add(self, text: str, user_id: str, metadata: dict[str, Any] | None = None) -> None:
        raise RuntimeError("add failed")

    def search(self, query: str, user_id: str, limit: int = 5) -> list[str]:
        raise RuntimeError("search failed")

    def remove(self, text: str, user_id: str) -> bool:
        raise RuntimeError("remove failed")


# ---------------------------------------------------------------------------
# _text_jaccard
# ---------------------------------------------------------------------------


class TestTextJaccard:
    def test_identical(self) -> None:
        assert _text_jaccard("hello world", "hello world") == 1.0

    def test_no_overlap(self) -> None:
        assert _text_jaccard("hello", "world") == 0.0

    def test_partial_overlap(self) -> None:
        score = _text_jaccard("hello world", "hello there")
        assert 0.0 < score < 1.0

    def test_empty(self) -> None:
        assert _text_jaccard("", "hello") == 0.0
        assert _text_jaccard("hello", "") == 0.0


# ---------------------------------------------------------------------------
# CompositeMemoryBackend
# ---------------------------------------------------------------------------


class TestCompositeAdd:
    """Tests for add()."""

    def test_add_writes_to_both(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        backend = CompositeMemoryBackend(p, s)
        backend.add("fact A", user_id="alice")
        assert p.added == [("fact A", "alice")]
        assert s.added == [("fact A", "alice")]

    def test_add_survives_primary_failure(self) -> None:
        p = FailingBackend()
        s = RecordingBackend()
        backend = CompositeMemoryBackend(p, s)
        backend.add("fact A", user_id="alice")
        assert s.added == [("fact A", "alice")]

    def test_add_survives_secondary_failure(self) -> None:
        p = RecordingBackend()
        s = FailingBackend()
        backend = CompositeMemoryBackend(p, s)
        backend.add("fact A", user_id="alice")
        assert p.added == [("fact A", "alice")]


class TestCompositeSearch:
    """Tests for search()."""

    def test_merges_results(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        p.search_results = ["fact A", "fact B"]
        s.search_results = ["fact C", "fact D"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice", limit=10)
        assert results == ["fact A", "fact B", "fact C", "fact D"]

    def test_deduplicates(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        p.search_results = ["dark mode preference"]
        s.search_results = ["dark mode preference"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice", limit=10)
        assert len(results) == 1

    def test_primary_first(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        p.search_results = ["primary result"]
        s.search_results = ["secondary result"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice")
        assert results[0] == "primary result"

    def test_respects_limit(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        p.search_results = ["a", "b", "c"]
        s.search_results = ["d", "e", "f"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice", limit=3)
        assert len(results) == 3

    def test_survives_primary_search_failure(self) -> None:
        p = FailingBackend()
        s = RecordingBackend()
        s.search_results = ["fact X"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice")
        assert results == ["fact X"]

    def test_survives_secondary_search_failure(self) -> None:
        p = RecordingBackend()
        s = FailingBackend()
        p.search_results = ["fact Y"]

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice")
        assert results == ["fact Y"]

    def test_empty_when_both_fail(self) -> None:
        p = FailingBackend()
        s = FailingBackend()

        backend = CompositeMemoryBackend(p, s)
        results = backend.search("query", user_id="alice")
        assert results == []


# ---------------------------------------------------------------------------
# remove()
# ---------------------------------------------------------------------------


class TestCompositeRemove:
    """Tests for remove()."""

    def test_remove_deletes_from_both(self) -> None:
        p = RecordingBackend()
        s = RecordingBackend()
        p.search_results = ["remember this"]
        s.search_results = ["remember this"]

        backend = CompositeMemoryBackend(p, s)
        removed = backend.remove("remember this", user_id="alice")

        assert removed is True
        assert p.search_results == []
        assert s.search_results == []

    def test_remove_survives_backend_failure(self) -> None:
        p = FailingBackend()
        s = RecordingBackend()
        s.search_results = ["remember this"]

        backend = CompositeMemoryBackend(p, s)
        removed = backend.remove("remember this", user_id="alice")

        assert removed is True
        assert s.search_results == []


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    def test_is_subclass(self) -> None:
        assert issubclass(CompositeMemoryBackend, MemoryBackend)

    def test_with_null_backends(self) -> None:
        backend = CompositeMemoryBackend(NullMemoryBackend(), NullMemoryBackend())
        backend.add("test", user_id="u")
        assert backend.search("test", user_id="u") == []
