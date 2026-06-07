"""Tests for agent/context.py."""

from miniclaw.agent.context import ContextManager, _rule_based_summarizer


# ============================================================
# add_message / get_messages / clear
# ============================================================


class TestMessageManagement:
    def test_add_message(self):
        ctx = ContextManager()
        ctx.add_message("user", "hello")
        assert ctx.message_count == 1
        assert ctx.get_messages() == [{"role": "user", "content": "hello"}]

    def test_add_multiple_roles(self):
        ctx = ContextManager()
        ctx.add_message("system", "You are helpful.")
        ctx.add_message("user", "Hi")
        ctx.add_message("assistant", "Hello!")
        ctx.add_message("tool", "result")
        ctx.add_message("summary", "compressed")
        assert ctx.message_count == 5
        roles = [m["role"] for m in ctx.get_messages()]
        assert roles == ["system", "user", "assistant", "tool", "summary"]

    def test_add_message_with_extra_fields(self):
        ctx = ContextManager()
        ctx.add_message("tool", "Sunny 25°C", tool_name="get_weather")
        msg = ctx.get_messages()[0]
        assert msg["tool_name"] == "get_weather"
        assert msg["content"] == "Sunny 25°C"

    def test_add_observation(self):
        ctx = ContextManager()
        ctx.add_observation("get_weather", {"city": "Beijing"}, "Sunny 25°C")
        msgs = ctx.get_messages()
        assert len(msgs) == 2
        # Assistant message describing the call
        assert msgs[0]["role"] == "assistant"
        assert "get_weather" in msgs[0]["content"]
        assert msgs[0]["tool_name"] == "get_weather"
        # Tool message with the result
        assert msgs[1]["role"] == "tool"
        assert "Sunny 25°C" in msgs[1]["content"]

    def test_clear(self):
        ctx = ContextManager()
        ctx.add_message("user", "a")
        ctx.add_message("user", "b")
        ctx.clear()
        assert ctx.message_count == 0
        assert ctx.get_messages() == []

    def test_get_messages_returns_copy(self):
        ctx = ContextManager()
        ctx.add_message("user", "x")
        msgs = ctx.get_messages()
        msgs.append({"role": "fake", "content": "y"})
        assert ctx.message_count == 1  # original unchanged

    def test_len(self):
        ctx = ContextManager()
        assert len(ctx) == 0
        ctx.add_message("user", "hi")
        assert len(ctx) == 1

    def test_repr(self):
        ctx = ContextManager(max_context_tokens=4000)
        ctx.add_message("user", "hello")
        r = repr(ctx)
        assert "ContextManager" in r
        assert "4000" in r


# ============================================================
# Token estimation
# ============================================================


class TestTokenEstimation:
    def test_empty_context(self):
        ctx = ContextManager()
        assert ctx.estimate_tokens() == 0

    def test_simple_estimation(self):
        ctx = ContextManager()
        # 8 chars → 8/4 = 2 tokens
        ctx.add_message("user", "12345678")
        assert ctx.estimate_tokens() == 2

    def test_multiple_messages(self):
        ctx = ContextManager()
        ctx.add_message("user", "abcdefgh")  # 8 chars → 2 tokens
        ctx.add_message("assistant", "ijklmnop")  # 8 chars → 2 tokens
        assert ctx.estimate_tokens() == 4

    def test_estimation_with_arguments(self):
        ctx = ContextManager()
        # content: 4 chars = 1 token, arguments: ~20 chars ≈ 5 tokens
        ctx.add_message("assistant", "call", arguments={"city": "Beijing"})
        tokens = ctx.estimate_tokens()
        assert tokens > 1  # includes arguments

    def test_custom_messages_estimation(self):
        ctx = ContextManager()
        msgs = [{"role": "user", "content": "1234"}]
        assert ctx.estimate_tokens(msgs) == 1


# ============================================================
# should_compress
# ============================================================


class TestShouldCompress:
    def test_under_budget(self):
        ctx = ContextManager(max_context_tokens=100)
        ctx.add_message("user", "short")
        assert ctx.should_compress() is False

    def test_over_budget(self):
        ctx = ContextManager(max_context_tokens=10)
        # 400 chars → 100 tokens > 10
        ctx.add_message("user", "x" * 400)
        assert ctx.should_compress() is True

    def test_exact_budget(self):
        ctx = ContextManager(max_context_tokens=10)
        # 40 chars → 10 tokens == 10
        ctx.add_message("user", "x" * 40)
        assert ctx.should_compress() is False


# ============================================================
# compress
# ============================================================


class TestCompress:
    def test_no_compression_when_under_budget(self):
        ctx = ContextManager(max_context_tokens=1000)
        ctx.add_message("user", "hello")
        result = ctx.compress()
        assert result is False
        assert ctx.message_count == 1

    def test_compression_reduces_message_count(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        # Add many messages that exceed budget
        for i in range(20):
            ctx.add_message("user", f"message {i} " * 10)
        old_count = ctx.message_count
        result = ctx.compress()
        assert result is True
        assert ctx.message_count < old_count

    def test_compression_preserves_recent(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=4)
        for i in range(20):
            ctx.add_message("user", f"msg {i} " * 10)
        ctx.compress()
        msgs = ctx.get_messages()
        # Last 4 original messages should be preserved
        recent = [m for m in msgs if m["role"] != "summary"]
        # The last messages should contain "msg 19"
        assert any("msg 19" in m.get("content", "") for m in recent)

    def test_compression_creates_summary(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        for i in range(20):
            ctx.add_message("user", f"message {i} " * 10)
        ctx.compress()
        msgs = ctx.get_messages()
        summary_msgs = [m for m in msgs if m["role"] == "summary"]
        assert len(summary_msgs) == 1
        assert "Compressed" in summary_msgs[0]["content"]

    def test_compression_preserves_system_messages(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        ctx.add_message("system", "You are helpful.")
        for i in range(20):
            ctx.add_message("user", f"msg {i} " * 10)
        ctx.compress()
        msgs = ctx.get_messages()
        system_msgs = [m for m in msgs if m["role"] == "system"]
        assert len(system_msgs) == 1
        assert system_msgs[0]["content"] == "You are helpful."

    def test_compression_preserves_order(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        ctx.add_message("system", "sys")
        for i in range(20):
            ctx.add_message("user", f"msg {i} " * 10)
        ctx.compress()
        msgs = ctx.get_messages()
        # system should be first
        assert msgs[0]["role"] == "system"
        # summary should come after system
        assert msgs[1]["role"] == "summary"
        # recent messages at the end
        assert msgs[-1]["role"] == "user"

    def test_compression_stays_over_budget_only_summary_and_recent(self):
        """After compression, only system + summary + recent_keep remain."""
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        ctx.add_message("system", "sys")
        for i in range(50):
            ctx.add_message("user", f"msg {i} " * 20)
        ctx.compress()
        msgs = ctx.get_messages()
        # system + summary + 2 recent = 4
        assert len(msgs) == 4

    def test_double_compression(self):
        """Compressing twice should be idempotent if still over budget."""
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        for i in range(20):
            ctx.add_message("user", f"msg {i} " * 10)
        ctx.compress()
        count1 = ctx.message_count
        # Compress again (may or may not compress further)
        ctx.compress()
        count2 = ctx.message_count
        assert count2 <= count1

    def test_custom_summarizer(self):
        def my_summarizer(msgs):
            return f"Custom summary of {len(msgs)} messages."

        ctx = ContextManager(max_context_tokens=10, recent_keep=2, summarizer=my_summarizer)
        for i in range(20):
            ctx.add_message("user", f"msg {i} " * 10)
        ctx.compress()
        msgs = ctx.get_messages()
        summary = [m for m in msgs if m["role"] == "summary"]
        assert "Custom summary" in summary[0]["content"]

    def test_compress_with_tool_messages(self):
        ctx = ContextManager(max_context_tokens=10, recent_keep=2)
        for i in range(10):
            ctx.add_observation(f"tool_{i}", {"arg": i}, f"result_{i} " * 10)
        old_count = ctx.message_count
        ctx.compress()
        assert ctx.message_count < old_count


# ============================================================
# build_messages
# ============================================================


class TestBuildMessages:
    def test_empty_build(self):
        ctx = ContextManager()
        msgs = ctx.build_messages()
        assert msgs == []

    def test_system_prompt_included(self):
        ctx = ContextManager()
        msgs = ctx.build_messages(system_prompt="You are helpful.")
        assert msgs[0]["role"] == "system"
        assert msgs[0]["content"] == "You are helpful."

    def test_tools_prompt_included(self):
        ctx = ContextManager()
        msgs = ctx.build_messages(tools_prompt="## Tools\n- echo")
        assert any("echo" in m["content"] for m in msgs)
        assert msgs[0]["role"] == "system"

    def test_long_term_memory_included(self):
        ctx = ContextManager()
        msgs = ctx.build_messages(long_term_memory="User prefers Chinese.")
        assert any("Chinese" in m["content"] for m in msgs)

    def test_current_task_is_last(self):
        ctx = ContextManager()
        ctx.add_message("user", "previous message")
        msgs = ctx.build_messages(current_task="What's the weather?")
        assert msgs[-1]["role"] == "user"
        assert msgs[-1]["content"] == "What's the weather?"

    def test_stored_messages_included(self):
        ctx = ContextManager()
        ctx.add_message("user", "hello")
        ctx.add_message("assistant", "hi there")
        msgs = ctx.build_messages(current_task="next question")
        # system (none) + 2 stored + current task
        contents = [m["content"] for m in msgs]
        assert "hello" in contents
        assert "hi there" in contents
        assert "next question" in contents

    def test_full_build_order(self):
        ctx = ContextManager()
        ctx.add_message("summary", "compressed old stuff")
        ctx.add_message("user", "recent msg")
        msgs = ctx.build_messages(
            system_prompt="sys",
            tools_prompt="tools",
            long_term_memory="mem",
            current_task="task",
        )
        roles = [m["role"] for m in msgs]
        # system, system(tools), system(memory), summary, user, user(task)
        assert roles[0] == "system"  # sys
        assert roles[1] == "system"  # tools
        assert roles[2] == "system"  # memory
        assert roles[3] == "summary"
        assert roles[4] == "user"  # recent msg
        assert roles[5] == "user"  # current task

    def test_no_optional_sections(self):
        ctx = ContextManager()
        ctx.add_message("assistant", "ok")
        msgs = ctx.build_messages()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "ok"


# ============================================================
# Rule-based summarizer
# ============================================================


class TestRuleBasedSummarizer:
    def test_empty_messages(self):
        assert _rule_based_summarizer([]) == ""

    def test_counts_roles(self):
        msgs = [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
            {"role": "user", "content": "bye"},
        ]
        summary = _rule_based_summarizer(msgs)
        assert "3 messages" in summary
        assert "user" in summary

    def test_tracks_tool_names(self):
        msgs = [
            {"role": "assistant", "content": "calling tool", "tool_name": "search"},
            {"role": "tool", "content": "result", "tool_name": "search"},
            {"role": "tool", "content": "result2", "tool_name": "calculator"},
        ]
        summary = _rule_based_summarizer(msgs)
        assert "search" in summary
        assert "calculator" in summary

    def test_includes_last_messages(self):
        msgs = [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "latest question"},
            {"role": "assistant", "content": "latest answer"},
        ]
        summary = _rule_based_summarizer(msgs)
        assert "latest question" in summary
        assert "latest answer" in summary

    def test_truncates_long_content(self):
        msgs = [{"role": "user", "content": "x" * 500}]
        summary = _rule_based_summarizer(msgs)
        # Should be truncated to 200 chars
        assert "xxx" in summary
        assert len(summary) < 400  # summary text is compact
