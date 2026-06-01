"""Tests for memory.py."""

import pytest
from miniclaw.memory import Memory


class TestMemoryKV:
    def test_save_and_load(self):
        mem = Memory(":memory:")
        mem.save("key1", {"a": 1})
        assert mem.load("key1") == {"a": 1}
        mem.close()

    def test_load_default(self):
        mem = Memory(":memory:")
        assert mem.load("missing") is None
        assert mem.load("missing", "fallback") == "fallback"
        mem.close()

    def test_overwrite(self):
        mem = Memory(":memory:")
        mem.save("k", "v1")
        mem.save("k", "v2")
        assert mem.load("k") == "v2"
        mem.close()

    def test_delete(self):
        mem = Memory(":memory:")
        mem.save("k", 1)
        assert mem.delete("k") is True
        assert mem.delete("k") is False
        assert mem.load("k") is None
        mem.close()

    def test_keys_with_prefix(self):
        mem = Memory(":memory:")
        mem.save("user:name", "Alice")
        mem.save("user:age", 30)
        mem.save("session:id", "abc")
        assert sorted(mem.keys("user:")) == ["user:age", "user:name"]
        assert mem.keys("session:") == ["session:id"]
        mem.close()


class TestMemoryMessages:
    def test_append_and_get(self):
        mem = Memory(":memory:")
        mem.append_message("user", "Hello")
        mem.append_message("assistant", "Hi there!")
        msgs = mem.get_messages()
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[1]["role"] == "assistant"
        mem.close()

    def test_limit(self):
        mem = Memory(":memory:")
        for i in range(10):
            mem.append_message("user", f"msg {i}")
        msgs = mem.get_messages(limit=3)
        assert len(msgs) == 3
        assert msgs[0]["content"] == "msg 7"
        assert msgs[2]["content"] == "msg 9"
        mem.close()

    def test_clear(self):
        mem = Memory(":memory:")
        mem.append_message("user", "bye")
        mem.clear_messages()
        assert mem.get_messages() == []
        mem.close()

    def test_metadata(self):
        mem = Memory(":memory:")
        mem.append_message("user", "hi", {"token_count": 5})
        msgs = mem.get_messages()
        assert msgs[0]["metadata"]["token_count"] == 5
        mem.close()

    def test_context_manager(self):
        with Memory(":memory:") as mem:
            mem.save("k", "v")
            assert mem.load("k") == "v"
