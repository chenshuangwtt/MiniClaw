"""Tests for memory/vector.py."""

import pytest

from miniclaw.memory.base import MemoryBackend
from miniclaw.memory.vector import VectorMemoryBackend, cosine_similarity, hash_embed


class TestVectorMemoryBackend:
    def test_is_memory_backend(self):
        backend = VectorMemoryBackend()
        assert isinstance(backend, MemoryBackend)

    def test_add_and_search_related_memory(self):
        backend = VectorMemoryBackend()
        backend.add("Alice prefers Python examples", user_id="alice")
        backend.add("Bob likes Rust examples", user_id="bob")

        results = backend.search("Python", user_id="alice")

        assert results == ["Alice prefers Python examples"]

    def test_search_is_user_scoped(self):
        backend = VectorMemoryBackend()
        backend.add("Alice likes coffee", user_id="alice")
        backend.add("Bob likes coffee", user_id="bob")

        assert backend.search("coffee", user_id="alice") == ["Alice likes coffee"]
        assert backend.search("coffee", user_id="bob") == ["Bob likes coffee"]

    def test_limit(self):
        backend = VectorMemoryBackend()
        backend.add("Python testing with pytest", user_id="alice")
        backend.add("Python package management with uv", user_id="alice")

        results = backend.search("Python", user_id="alice", limit=1)

        assert len(results) == 1

    def test_empty_text_is_ignored(self):
        backend = VectorMemoryBackend()
        backend.add("   ", user_id="alice")

        assert backend.entries() == []

    def test_empty_query_returns_empty(self):
        backend = VectorMemoryBackend()
        backend.add("Alice likes Python", user_id="alice")

        assert backend.search("", user_id="alice") == []

    def test_custom_embedder(self):
        calls: list[str] = []

        def embedder(text: str) -> list[float]:
            calls.append(text)
            return [1.0, 0.0] if "python" in text.lower() else [0.0, 1.0]

        backend = VectorMemoryBackend(embedder=embedder, dimensions=2)
        backend.add("Python memory", user_id="alice")
        backend.add("Rust memory", user_id="alice")

        results = backend.search("python", user_id="alice")

        assert results[0] == "Python memory"
        assert calls == ["Python memory", "Rust memory", "python"]

    def test_invalid_dimensions_raise(self):
        with pytest.raises(ValueError):
            VectorMemoryBackend(dimensions=0)


class TestVectorHelpers:
    def test_hash_embed_normalizes_vector(self):
        vector = hash_embed("Python Python testing", dimensions=16)
        norm = sum(value * value for value in vector) ** 0.5
        assert norm == pytest.approx(1.0)

    def test_cosine_similarity_identical_vectors(self):
        assert cosine_similarity([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_cosine_similarity_mismatched_dimensions(self):
        assert cosine_similarity([1.0], [1.0, 0.0]) == 0.0
