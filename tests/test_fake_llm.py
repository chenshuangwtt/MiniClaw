"""Tests for llm/fake.py."""

import pytest
from miniclaw.llm.fake import FakeLLM
from miniclaw.llm.base import LLMResponse


class TestFakeLLM:
    def test_sequential_responses(self):
        llm = FakeLLM(["First", "Second", "Third"])
        assert llm.chat([]).content == "First"
        assert llm.chat([]).content == "Second"
        assert llm.chat([]).content == "Third"

    def test_exhausted_returns_fallback(self):
        llm = FakeLLM(["Only one"])
        llm.chat([])
        resp = llm.chat([])
        assert "No more scripted responses" in resp.content

    def test_exhausted_property(self):
        llm = FakeLLM(["one"])
        assert not llm.exhausted
        llm.chat([])
        assert llm.exhausted

    def test_reset(self):
        llm = FakeLLM(["A", "B"])
        llm.chat([])
        llm.chat([])
        assert llm.exhausted
        llm.reset()
        assert not llm.exhausted
        assert llm.chat([]).content == "A"

    def test_tool_call_parsing(self):
        llm = FakeLLM([
            '{"tool_call": {"name": "search", "arguments": {"q": "test"}}}'
        ])
        resp = llm.chat([])
        assert resp.has_tool_calls
        assert resp.tool_calls[0].name == "search"
        assert resp.tool_calls[0].arguments == {"q": "test"}

    def test_llm_response_passthrough(self):
        direct = LLMResponse(content="direct", tool_calls=[])
        llm = FakeLLM([direct])
        resp = llm.chat([])
        assert resp.content == "direct"

    def test_call_log(self):
        llm = FakeLLM(["ok"])
        llm.chat([{"role": "user", "content": "hi"}], tools=[{"type": "function"}])
        assert len(llm.call_log) == 1
        assert llm.call_log[0]["temperature"] == 0.0

    def test_empty_response(self):
        llm = FakeLLM([""])
        resp = llm.chat([])
        assert resp.content == ""
