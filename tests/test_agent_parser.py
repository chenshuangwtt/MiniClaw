"""Tests for agent/state.py and agent/parser.py."""

import pytest
from pydantic import ValidationError

from miniclaw.agent.state import FinalAnswer, ToolCall
from miniclaw.agent.parser import OutputParser, ParseError


# ============================================================
# Pydantic model tests (state.py)
# ============================================================


class TestToolCallModel:
    def test_minimal(self):
        tc = ToolCall(tool_name="search")
        assert tc.type == "tool_call"
        assert tc.tool_name == "search"
        assert tc.arguments == {}
        assert tc.thought == ""

    def test_full(self):
        tc = ToolCall(
            thought="I need to search.",
            tool_name="search",
            arguments={"q": "python"},
        )
        assert tc.thought == "I need to search."
        assert tc.arguments == {"q": "python"}

    def test_from_dict(self):
        data = {
            "type": "tool_call",
            "thought": "Let me check.",
            "tool_name": "get_weather",
            "arguments": {"city": "Beijing"},
        }
        tc = ToolCall.model_validate(data)
        assert tc.tool_name == "get_weather"
        assert tc.arguments == {"city": "Beijing"}

    def test_missing_tool_name_raises(self):
        with pytest.raises(ValidationError):
            ToolCall.model_validate({"type": "tool_call"})

    def test_default_type_field(self):
        tc = ToolCall(tool_name="x")
        assert tc.type == "tool_call"


class TestFinalAnswerModel:
    def test_minimal(self):
        fa = FinalAnswer(answer="Hello!")
        assert fa.type == "final_answer"
        assert fa.answer == "Hello!"
        assert fa.thought == ""

    def test_full(self):
        fa = FinalAnswer(
            thought="I have all the info.",
            answer="The answer is 42.",
        )
        assert fa.thought == "I have all the info."
        assert fa.answer == "The answer is 42."

    def test_from_dict(self):
        data = {
            "type": "final_answer",
            "thought": "Done.",
            "answer": "It's sunny.",
        }
        fa = FinalAnswer.model_validate(data)
        assert fa.answer == "It's sunny."

    def test_missing_answer_raises(self):
        with pytest.raises(ValidationError):
            FinalAnswer.model_validate({"type": "final_answer"})


# ============================================================
# Parser tests (parser.py)
# ============================================================


class TestOutputParser:
    def setup_method(self):
        self.parser = OutputParser()

    # --- Successful parsing ---

    def test_parse_final_answer(self):
        raw = '{"type": "final_answer", "thought": "", "answer": "Hi!"}'
        result = self.parser.parse(raw)
        assert isinstance(result, FinalAnswer)
        assert result.answer == "Hi!"

    def test_parse_tool_call(self):
        raw = '{"type": "tool_call", "thought": "Need info.", "tool_name": "search", "arguments": {"q": "test"}}'
        result = self.parser.parse(raw)
        assert isinstance(result, ToolCall)
        assert result.tool_name == "search"
        assert result.arguments == {"q": "test"}

    def test_parse_tool_call_no_arguments(self):
        raw = '{"type": "tool_call", "tool_name": "ping"}'
        result = self.parser.parse(raw)
        assert isinstance(result, ToolCall)
        assert result.arguments == {}

    def test_parse_final_answer_no_thought(self):
        raw = '{"type": "final_answer", "answer": "ok"}'
        result = self.parser.parse(raw)
        assert isinstance(result, FinalAnswer)
        assert result.thought == ""

    # --- Error cases ---

    def test_empty_string_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse("")

    def test_whitespace_only_raises(self):
        with pytest.raises(ParseError, match="Empty"):
            self.parser.parse("   ")

    def test_invalid_json_raises(self):
        with pytest.raises(ParseError, match="Invalid JSON"):
            self.parser.parse("{broken json")

    def test_non_dict_json_raises(self):
        with pytest.raises(ParseError, match="Expected a JSON object"):
            self.parser.parse("[1, 2, 3]")

    def test_missing_type_field_raises(self):
        with pytest.raises(ParseError, match='Missing required field "type"'):
            self.parser.parse('{"answer": "no type"}')

    def test_unknown_type_raises(self):
        with pytest.raises(ParseError, match="Unknown type"):
            self.parser.parse('{"type": "unknown_thing", "data": 1}')

    def test_validation_error_raises(self):
        # type=tool_call but missing required tool_name
        with pytest.raises(ParseError, match="Validation error"):
            self.parser.parse('{"type": "tool_call", "arguments": {}}')

    def test_parse_error_preserves_raw(self):
        try:
            self.parser.parse("{bad")
        except ParseError as exc:
            assert exc.raw == "{bad"
            assert "Invalid JSON" in exc.detail

    # --- Integration with both types ---

    def test_discriminated_union(self):
        """Ensure the parser correctly routes by the 'type' field."""
        tool_raw = '{"type": "tool_call", "tool_name": "calc", "arguments": {"expr": "1+1"}}'
        answer_raw = '{"type": "final_answer", "answer": "42"}'

        tool_result = self.parser.parse(tool_raw)
        answer_result = self.parser.parse(answer_raw)

        assert isinstance(tool_result, ToolCall)
        assert isinstance(answer_result, FinalAnswer)
        assert not isinstance(tool_result, FinalAnswer)
        assert not isinstance(answer_result, ToolCall)
