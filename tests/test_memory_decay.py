"""Tests for memory decay on VectorMemoryBackend and SQLiteStore."""

from __future__ import annotations

import pytest

from miniclaw.memory.vector import VectorMemoryBackend
from miniclaw.storage.sqlite_store import SQLiteStore


# ---------------------------------------------------------------------------
# VectorMemoryBackend decay
# ---------------------------------------------------------------------------


class TestVectorDecay:
    """Tests for VectorMemoryBackend.decay()."""

    def test_decay_reduces_importance(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice", metadata={"importance": 5.0})
        # Manually set importance high
        backend._entries[0].importance = 5.0

        decayed = backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert decayed == 1
        assert backend._entries[0].importance == pytest.approx(2.5)

    def test_decay_respects_min_importance(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice")
        backend._entries[0].importance = 0.2

        decayed = backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert decayed == 1
        assert backend._entries[0].importance == pytest.approx(0.1)

    def test_decay_does_not_go_below_min(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice")
        backend._entries[0].importance = 0.15

        decayed = backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert decayed == 1
        assert backend._entries[0].importance == pytest.approx(0.1)

    def test_decay_skips_already_at_min(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice")
        backend._entries[0].importance = 0.1

        decayed = backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert decayed == 0

    def test_decay_only_affects_target_user(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice")
        backend.add("fact B", user_id="bob")
        backend._entries[0].importance = 5.0
        backend._entries[1].importance = 5.0

        decayed = backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert decayed == 1
        assert backend._entries[0].importance == pytest.approx(2.5)
        assert backend._entries[1].importance == pytest.approx(5.0)  # untouched

    def test_decay_updates_timestamp(self) -> None:
        backend = VectorMemoryBackend()
        backend.add("fact A", user_id="alice")
        backend._entries[0].importance = 5.0
        old_ts = backend._entries[0].updated_at

        import time

        time.sleep(0.01)
        backend.decay("alice", decay_factor=0.5, min_importance=0.1)
        assert backend._entries[0].updated_at >= old_ts

    def test_decay_empty_backend(self) -> None:
        backend = VectorMemoryBackend()
        assert backend.decay("alice") == 0


# ---------------------------------------------------------------------------
# SQLiteStore decay
# ---------------------------------------------------------------------------


class TestSQLiteDecay:
    """Tests for SQLiteStore.decay()."""

    def test_decay_reduces_importance(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=10, user_id="alice")
            store.save_memory("k2", "v2", importance=5, user_id="alice")

            decayed = store.decay("alice", decay_factor=0.5, min_importance=1)
            assert decayed == 2

            mems = store.list_memories(user_id="alice")
            by_key = {m["key"]: m for m in mems}
            assert by_key["k1"]["importance"] == 5
            assert by_key["k2"]["importance"] == 2  # 5*0.5=2.5 → int = 2

    def test_decay_respects_min_importance(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=2, user_id="alice")

            decayed = store.decay("alice", decay_factor=0.5, min_importance=1)
            assert decayed == 1

            mems = store.list_memories(user_id="alice")
            assert mems[0]["importance"] == 1  # MAX(1, 2*0.5=1) = 1

    def test_decay_skips_already_at_min(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=1, user_id="alice")

            decayed = store.decay("alice", decay_factor=0.5, min_importance=1)
            assert decayed == 0

    def test_decay_only_affects_target_user(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=10, user_id="alice")
            store.save_memory("k2", "v2", importance=10, user_id="bob")

            decayed = store.decay("alice", decay_factor=0.5, min_importance=1)
            assert decayed == 1

            alice_mems = store.list_memories(user_id="alice")
            bob_mems = store.list_memories(user_id="bob")
            assert alice_mems[0]["importance"] == 5
            assert bob_mems[0]["importance"] == 10  # untouched

    def test_decay_updates_timestamp(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=10, user_id="alice")

            before = store.list_memories(user_id="alice")[0]["updated_at"]
            store.decay("alice", decay_factor=0.5, min_importance=1)
            after = store.list_memories(user_id="alice")[0]["updated_at"]
            # updated_at should be >= before (same or later)
            assert after >= before

    def test_decay_default_user(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=10)

            decayed = store.decay()  # default user_id="default"
            assert decayed == 1

    def test_save_memory_with_user_id(self) -> None:
        with SQLiteStore(":memory:") as store:
            store.save_memory("k1", "v1", importance=5, user_id="alice")
            mems = store.list_memories(user_id="alice")
            assert len(mems) == 1
            assert mems[0]["user_id"] == "alice"
