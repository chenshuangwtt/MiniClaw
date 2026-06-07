"""Tests for memory/mem0_store.py — using mocks, no real Mem0 calls."""

from unittest.mock import MagicMock, patch

from miniclaw.memory.base import MemoryBackend
from miniclaw.memory.mem0_store import Mem0MemoryBackend, _extract_texts


# ============================================================
# _extract_texts — result normalization
# ============================================================


class TestExtractTexts:
    def test_none_returns_empty(self):
        assert _extract_texts(None) == []

    def test_list_of_dicts_with_memory_key(self):
        results = [
            {"memory": "Alice likes coffee", "score": 0.95},
            {"memory": "Alice prefers Python", "score": 0.87},
        ]
        assert _extract_texts(results) == ["Alice likes coffee", "Alice prefers Python"]

    def test_list_of_dicts_with_text_key(self):
        results = [{"text": "hello"}, {"text": "world"}]
        assert _extract_texts(results) == ["hello", "world"]

    def test_list_of_dicts_with_content_key(self):
        results = [{"content": "fact1"}, {"content": "fact2"}]
        assert _extract_texts(results) == ["fact1", "fact2"]

    def test_list_of_dicts_with_value_key(self):
        results = [{"value": "v1"}]
        assert _extract_texts(results) == ["v1"]

    def test_list_of_strings(self):
        results = ["memory1", "memory2"]
        assert _extract_texts(results) == ["memory1", "memory2"]

    def test_dict_with_results_key(self):
        results = {"results": [{"memory": "a"}, {"memory": "b"}]}
        assert _extract_texts(results) == ["a", "b"]

    def test_dict_with_memories_key(self):
        results = {"memories": ["x", "y"]}
        assert _extract_texts(results) == ["x", "y"]

    def test_single_string(self):
        assert _extract_texts("just one") == ["just one"]

    def test_empty_list(self):
        assert _extract_texts([]) == []

    def test_empty_dict(self):
        assert _extract_texts({}) == []

    def test_mixed_types(self):
        results = ["plain", {"memory": "dict"}, 42]
        texts = _extract_texts(results)
        assert "plain" in texts
        assert "dict" in texts
        assert "42" in texts

    def test_dict_with_no_known_keys(self):
        results = [{"unknown": "value"}]
        assert _extract_texts(results) == []

    def test_non_list_non_dict(self):
        assert _extract_texts(42) == []


# ============================================================
# Mem0MemoryBackend — with mocked Mem0
# ============================================================


class TestMem0MemoryBackendInit:
    def test_is_subclass(self):
        assert issubclass(Mem0MemoryBackend, MemoryBackend)

    @patch("miniclaw.memory.mem0_store._init_mem0", return_value=None)
    def test_unavailable_backend(self, mock_init):
        backend = Mem0MemoryBackend()
        assert backend.is_available is False

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_available_backend(self, mock_init):
        mock_init.return_value = MagicMock()
        backend = Mem0MemoryBackend()
        assert backend.is_available is True


class TestMem0MemoryBackendAdd:
    @patch("miniclaw.memory.mem0_store._init_mem0", return_value=None)
    def test_add_when_unavailable(self, mock_init):
        """Should not crash when Mem0 is unavailable."""
        backend = Mem0MemoryBackend()
        backend.add("test", "user1")  # Should silently do nothing

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_add_delegates_to_mem0(self, mock_init):
        mock_mem = MagicMock()
        mock_init.return_value = mock_mem
        backend = Mem0MemoryBackend()

        backend.add("remember this", user_id="alice", metadata={"source": "chat"})

        mock_mem.add.assert_called_once_with(
            "remember this", user_id="alice", metadata={"source": "chat"}
        )

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_add_handles_exception(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.add.side_effect = RuntimeError("connection failed")
        mock_init.return_value = mock_mem

        backend = Mem0MemoryBackend()
        backend.add("test", "user1")  # Should not raise


class TestMem0MemoryBackendSearch:
    @patch("miniclaw.memory.mem0_store._init_mem0", return_value=None)
    def test_search_when_unavailable(self, mock_init):
        backend = Mem0MemoryBackend()
        assert backend.search("query", "user1") == []

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_search_delegates_to_mem0(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.search.return_value = [
            {"memory": "Alice likes Python", "score": 0.9},
        ]
        mock_init.return_value = mock_mem
        backend = Mem0MemoryBackend()

        results = backend.search("Alice preferences", user_id="alice")

        mock_mem.search.assert_called_once_with("Alice preferences", user_id="alice", limit=5)
        assert results == ["Alice likes Python"]

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_search_handles_exception(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.search.side_effect = RuntimeError("timeout")
        mock_init.return_value = mock_mem

        backend = Mem0MemoryBackend()
        assert backend.search("query", "user1") == []

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_search_with_limit(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.search.return_value = []
        mock_init.return_value = mock_mem
        backend = Mem0MemoryBackend()

        backend.search("q", "user1", limit=10)
        mock_mem.search.assert_called_once_with("q", user_id="user1", limit=10)

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_search_normalizes_string_results(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.search.return_value = ["fact1", "fact2"]
        mock_init.return_value = mock_mem

        backend = Mem0MemoryBackend()
        results = backend.search("q", "user1")
        assert results == ["fact1", "fact2"]

    @patch("miniclaw.memory.mem0_store._init_mem0")
    def test_search_normalizes_dict_results(self, mock_init):
        mock_mem = MagicMock()
        mock_mem.search.return_value = {"results": [{"memory": "a"}, {"memory": "b"}]}
        mock_init.return_value = mock_mem

        backend = Mem0MemoryBackend()
        results = backend.search("q", "user1")
        assert results == ["a", "b"]


# ============================================================
# _init_mem0 — import failure
# ============================================================


class TestInitMem0:
    @patch.dict("sys.modules", {"mem0": None})
    def test_import_failure_returns_none(self):
        from miniclaw.memory.mem0_store import _init_mem0

        assert _init_mem0(None) is None
