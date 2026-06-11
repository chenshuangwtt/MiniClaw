"""Tests for storage/sqlite_store.py."""

import json
from pathlib import Path


from miniclaw.storage.sqlite_store import SQLiteStore


# ============================================================
# Lifecycle
# ============================================================


class TestLifecycle:
    def test_context_manager(self, tmp_path: Path):
        db = tmp_path / "test.db"
        with SQLiteStore(db) as store:
            store.create_session("test")
        assert db.exists()

    def test_in_memory(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session("mem")
            assert sid >= 1

    def test_connect_and_close(self, tmp_path: Path):
        db = tmp_path / "test.db"
        store = SQLiteStore(db)
        store.connect()
        store.init_db()
        store.create_session("manual")
        store.close()

    def test_double_init_db_safe(self):
        with SQLiteStore(":memory:") as store:
            store.init_db()
            store.init_db()  # should not raise
            store.create_session("ok")

    def test_auto_connect_on_use(self):
        """Operations should auto-connect if not explicitly connected."""
        store = SQLiteStore(":memory:")
        sid = store.create_session("auto")
        assert sid >= 1
        store.close()


# ============================================================
# Sessions
# ============================================================


class TestSessions:
    def test_create_session_returns_id(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session("My Session")
            assert isinstance(sid, int)
            assert sid >= 1

    def test_create_session_default_title(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            assert sid >= 1

    def test_session_ids_increment(self):
        with SQLiteStore(":memory:") as store:
            s1 = store.create_session("A")
            s2 = store.create_session("B")
            assert s2 > s1


# ============================================================
# Messages
# ============================================================


class TestMessages:
    def test_save_and_list(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            store.save_message(sid, "user", "Hello")
            store.save_message(sid, "assistant", "Hi!")
            msgs = store.list_messages(sid)
            assert len(msgs) == 2
            assert msgs[0]["role"] == "user"
            assert msgs[0]["content"] == "Hello"
            assert msgs[1]["role"] == "assistant"

    def test_messages_ordered_by_id(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            for i in range(5):
                store.save_message(sid, "user", f"msg {i}")
            msgs = store.list_messages(sid)
            contents = [m["content"] for m in msgs]
            assert contents == ["msg 0", "msg 1", "msg 2", "msg 3", "msg 4"]

    def test_messages_limited(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            for i in range(10):
                store.save_message(sid, "user", f"msg {i}")
            msgs = store.list_messages(sid, limit=3)
            assert len(msgs) == 3

    def test_messages_isolated_by_session(self):
        with SQLiteStore(":memory:") as store:
            s1 = store.create_session("A")
            s2 = store.create_session("B")
            store.save_message(s1, "user", "in session A")
            store.save_message(s2, "user", "in session B")
            msgs1 = store.list_messages(s1)
            msgs2 = store.list_messages(s2)
            assert len(msgs1) == 1
            assert len(msgs2) == 1
            assert msgs1[0]["content"] == "in session A"
            assert msgs2[0]["content"] == "in session B"

    def test_message_has_created_at(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            store.save_message(sid, "user", "hi")
            msgs = store.list_messages(sid)
            assert "created_at" in msgs[0]
            assert msgs[0]["created_at"]  # not empty

    def test_save_message_returns_id(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            mid = store.save_message(sid, "user", "test")
            assert isinstance(mid, int)
            assert mid >= 1


# ============================================================
# Memories
# ============================================================


class TestMemories:
    def test_save_and_list(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("user:name", "Alice", importance=5)
            store.save_memory("user:age", "30", importance=3)
            mems = store.list_memories()
            assert len(mems) == 2
            # Ordered by importance DESC
            assert mems[0]["key"] == "user:name"
            assert mems[0]["importance"] == 5
            assert mems[1]["key"] == "user:age"

    def test_save_memory_returns_id(self):
        with SQLiteStore(":memory:") as store:
            mid = store.save_memory("k", "v")
            assert isinstance(mid, int)
            assert mid >= 1

    def test_upsert_on_duplicate_key(self):
        with SQLiteStore(":memory:") as store:
            mid1 = store.save_memory("k", "v1", importance=1)
            mid2 = store.save_memory("k", "v2", importance=9)
            assert mid1 == mid2  # same row
            mems = store.list_memories()
            assert len(mems) == 1
            assert mems[0]["value"] == "v2"
            assert mems[0]["importance"] == 9

    def test_same_key_is_isolated_by_user_id(self):
        with SQLiteStore(":memory:") as store:
            alice_id = store.save_memory("pref:theme", "dark", user_id="alice")
            bob_id = store.save_memory("pref:theme", "light", user_id="bob")

            assert alice_id != bob_id
            alice_mems = store.list_memories(user_id="alice")
            bob_mems = store.list_memories(user_id="bob")
            assert alice_mems[0]["value"] == "dark"
            assert bob_mems[0]["value"] == "light"

    def test_default_importance(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("k", "v")
            mems = store.list_memories()
            assert mems[0]["importance"] == 1

    def test_list_limited(self):
        with SQLiteStore(":memory:") as store:
            for i in range(10):
                store.save_memory(f"k{i}", f"v{i}")
            mems = store.list_memories(limit=3)
            assert len(mems) == 3

    def test_memory_has_timestamps(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("k", "v")
            mems = store.list_memories()
            assert mems[0]["created_at"]
            assert mems[0]["updated_at"]


class TestSearchMemories:
    def test_search_by_key(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("user:name", "Alice")
            store.save_memory("user:city", "Beijing")
            store.save_memory("project", "MiniClaw")
            results = store.search_memories("user")
            assert len(results) == 2
            keys = {r["key"] for r in results}
            assert keys == {"user:name", "user:city"}

    def test_search_by_value(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("name", "Alice")
            store.save_memory("city", "Beijing")
            store.save_memory("pet", "Alice the cat")
            results = store.search_memories("Alice")
            assert len(results) == 2

    def test_search_case_insensitive(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("Name", "Alice")
            results = store.search_memories("alice")
            # SQLite LIKE is case-insensitive by default for ASCII
            assert len(results) >= 1

    def test_search_no_results(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("k", "v")
            results = store.search_memories("nonexistent")
            assert len(results) == 0

    def test_search_limited(self):
        with SQLiteStore(":memory:") as store:
            for i in range(10):
                store.save_memory(f"match_{i}", f"value_{i}")
            results = store.search_memories("match", limit=3)
            assert len(results) == 3

    def test_search_ordered_by_importance(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("low", "findme", importance=1)
            store.save_memory("high", "findme", importance=10)
            store.save_memory("mid", "findme", importance=5)
            results = store.search_memories("findme")
            importances = [r["importance"] for r in results]
            assert importances == [10, 5, 1]

    def test_search_can_filter_by_user_id(self):
        with SQLiteStore(":memory:") as store:
            store.save_memory("pref:theme", "dark", user_id="alice")
            store.save_memory("pref:theme", "light", user_id="bob")

            results = store.search_memories("pref", user_id="alice")
            assert len(results) == 1
            assert results[0]["user_id"] == "alice"
            assert results[0]["value"] == "dark"


# ============================================================
# Traces
# ============================================================


class TestTraces:
    def test_save_and_list(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            store.save_trace(sid, step=1, event_json='{"action": "tool_call"}')
            store.save_trace(sid, step=2, event_json='{"action": "final_answer"}')
            traces = store.list_traces(sid)
            assert len(traces) == 2
            assert traces[0]["step"] == 1
            assert traces[1]["step"] == 2

    def test_traces_ordered_by_step(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            store.save_trace(sid, step=3, event_json='{"s": 3}')
            store.save_trace(sid, step=1, event_json='{"s": 1}')
            store.save_trace(sid, step=2, event_json='{"s": 2}')
            traces = store.list_traces(sid)
            steps = [t["step"] for t in traces]
            assert steps == [1, 2, 3]

    def test_traces_limited(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            for i in range(10):
                store.save_trace(sid, step=i, event_json=f'{{"s": {i}}}')
            traces = store.list_traces(sid, limit=5)
            assert len(traces) == 5

    def test_traces_isolated_by_session(self):
        with SQLiteStore(":memory:") as store:
            s1 = store.create_session()
            s2 = store.create_session()
            store.save_trace(s1, step=1, event_json='{"s": "a"}')
            store.save_trace(s2, step=1, event_json='{"s": "b"}')
            t1 = store.list_traces(s1)
            t2 = store.list_traces(s2)
            assert len(t1) == 1
            assert len(t2) == 1
            assert "a" in t1[0]["event_json"]
            assert "b" in t2[0]["event_json"]

    def test_trace_has_created_at(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            store.save_trace(sid, step=1, event_json="{}")
            traces = store.list_traces(sid)
            assert traces[0]["created_at"]

    def test_save_trace_returns_id(self):
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            tid = store.save_trace(sid, step=1, event_json="{}")
            assert isinstance(tid, int)
            assert tid >= 1

    def test_trace_json_roundtrip(self):
        """JSON stored in trace can be parsed back."""
        with SQLiteStore(":memory:") as store:
            sid = store.create_session()
            event = {"step": 1, "tool": "echo", "result": "hello"}
            store.save_trace(sid, step=1, event_json=json.dumps(event))
            traces = store.list_traces(sid)
            parsed = json.loads(traces[0]["event_json"])
            assert parsed["tool"] == "echo"
            assert parsed["result"] == "hello"


# ============================================================
# File-based persistence
# ============================================================


class TestFilePersistence:
    def test_data_persists_across_connections(self, tmp_path: Path):
        db = tmp_path / "persist.db"

        # Write
        with SQLiteStore(db) as store:
            sid = store.create_session("test")
            store.save_message(sid, "user", "hello")
            store.save_memory("k", "v")

        # Read in new connection
        with SQLiteStore(db) as store:
            msgs = store.list_messages(sid)
            assert len(msgs) == 1
            assert msgs[0]["content"] == "hello"
            mems = store.list_memories()
            assert len(mems) == 1
            assert mems[0]["value"] == "v"
