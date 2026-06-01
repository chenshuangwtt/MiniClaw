"""Tests for json_parser.py."""

import pytest
from miniclaw.json_parser import parse_llm_response


class TestParseLLMResponse:
    def test_plain_text(self):
        result = parse_llm_response("Hello, how can I help?")
        assert result.text == "Hello, how can I help?"
        assert not result.has_tool_calls

    def test_empty_string(self):
        result = parse_llm_response("")
        assert result.text == ""
        assert not result.has_tool_calls

    def test_whole_json_tool_call(self):
        raw = '{"tool_call": {"name": "get_weather", "arguments": {"city": "Beijing"}}}'
        result = parse_llm_response(raw)
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "get_weather"
        assert result.tool_calls[0].arguments == {"city": "Beijing"}

    def test_whole_json_with_content(self):
        raw = '{"content": "Let me check.", "tool_call": {"name": "search", "arguments": {"q": "test"}}}'
        result = parse_llm_response(raw)
        assert result.has_tool_calls
        assert result.text == "Let me check."

    def test_code_fence_json(self):
        raw = 'Here you go:\n```json\n{"tool_call": {"name": "calc", "arguments": {"expr": "1+1"}}}\n```'
        result = parse_llm_response(raw)
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "calc"

    def test_embedded_tool_call(self):
        raw = 'I will search for you. {"tool_call": {"name": "search", "arguments": {"q": "python"}}} Done.'
        result = parse_llm_response(raw)
        assert result.has_tool_calls
        assert result.tool_calls[0].name == "search"

    def test_multiple_tool_calls(self):
        raw = '{"tool_calls": [{"name": "a", "arguments": {}}, {"name": "b", "arguments": {"x": 1}}]}'
        result = parse_llm_response(raw)
        assert len(result.tool_calls) == 2
        assert result.tool_calls[0].name == "a"
        assert result.tool_calls[1].name == "b"

    def test_invalid_json_returns_text(self):
        raw = "This is not JSON at all {broken"
        result = parse_llm_response(raw)
        assert result.text == raw
        assert not result.has_tool_calls

    def test_json_array_returns_text(self):
        raw = "[1, 2, 3]"
        result = parse_llm_response(raw)
        assert not result.has_tool_calls
