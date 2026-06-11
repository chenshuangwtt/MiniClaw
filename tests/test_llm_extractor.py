"""Tests for LLMMemoryExtractor."""

from __future__ import annotations


from miniclaw.memory.extractor import LLMMemoryExtractor, MemoryExtractor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeLLM:
    """Minimal fake LLM that returns scripted responses."""

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self._index = 0

    def generate(self, prompt: str) -> str:
        if self._index >= len(self._responses):
            raise RuntimeError("FakeLLM exhausted")
        resp = self._responses[self._index]
        self._index += 1
        return resp


class FailingLLM:
    """LLM that always raises."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM unavailable")


# ---------------------------------------------------------------------------
# should_remember
# ---------------------------------------------------------------------------


class TestShouldRemember:
    """Tests for LLMMemoryExtractor.should_remember."""

    def test_llm_says_yes(self) -> None:
        llm = FakeLLM(["yes"])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.should_remember("I prefer dark mode") is True

    def test_llm_says_no(self) -> None:
        llm = FakeLLM(["no"])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.should_remember("What time is it?") is False

    def test_llm_says_yes_with_prefix(self) -> None:
        llm = FakeLLM(["Yes, this is worth remembering."])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.should_remember("Remember this") is True

    def test_sensitive_blocked(self) -> None:
        llm = FakeLLM(["yes"])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.should_remember("password=abc123") is False

    def test_fallback_on_error(self) -> None:
        llm = FailingLLM()
        extractor = LLMMemoryExtractor(llm, keywords=["记住"])
        # Should fall back to keyword-based
        assert extractor.should_remember("请记住我的偏好") is True
        assert extractor.should_remember("random text") is False

    def test_no_fallback_on_error(self) -> None:
        llm = FailingLLM()
        extractor = LLMMemoryExtractor(llm, fallback_on_error=False)
        assert extractor.should_remember("请记住我的偏好") is False


# ---------------------------------------------------------------------------
# extract
# ---------------------------------------------------------------------------


class TestExtract:
    """Tests for LLMMemoryExtractor.extract."""

    def test_extract_json_array(self) -> None:
        llm = FakeLLM(['["I prefer dark mode", "My name is Bob"]'])
        extractor = LLMMemoryExtractor(llm)
        result = extractor.extract("I prefer dark mode, and my name is Bob")
        assert result == ["I prefer dark mode", "My name is Bob"]

    def test_extract_empty_array(self) -> None:
        llm = FakeLLM(["[]"])
        extractor = LLMMemoryExtractor(llm)
        result = extractor.extract("What's the weather?")
        assert result == []

    def test_extract_with_code_fence(self) -> None:
        llm = FakeLLM(['```json\n["fact 1", "fact 2"]\n```'])
        extractor = LLMMemoryExtractor(llm)
        result = extractor.extract("some text")
        assert result == ["fact 1", "fact 2"]

    def test_extract_filters_sensitive(self) -> None:
        llm = FakeLLM(['["My API key is sk-abcdefghijklmnopqrstuvwx"]'])
        extractor = LLMMemoryExtractor(llm)
        result = extractor.extract("some text")
        assert result == []

    def test_extract_empty_text(self) -> None:
        llm = FakeLLM(["[]"])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.extract("") == []
        assert extractor.extract("   ") == []

    def test_extract_sensitive_text_blocked(self) -> None:
        llm = FakeLLM(['["should not reach here"]'])
        extractor = LLMMemoryExtractor(llm)
        assert extractor.extract("token=secret123") == []

    def test_extract_fallback_on_error(self) -> None:
        llm = FailingLLM()
        extractor = LLMMemoryExtractor(llm, keywords=["记住"])
        result = extractor.extract("请记住我喜欢蓝色")
        # Falls back to keyword-based, which splits by sentence and checks keywords
        assert len(result) > 0

    def test_extract_malformed_json_returns_empty(self) -> None:
        llm = FakeLLM(["not valid json at all"])
        extractor = LLMMemoryExtractor(llm)
        result = extractor.extract("some text")
        assert result == []


# ---------------------------------------------------------------------------
# _parse_json_array
# ---------------------------------------------------------------------------


class TestParseJsonArray:
    """Tests for the JSON parsing helper."""

    def test_direct_array(self) -> None:
        assert LLMMemoryExtractor._parse_json_array('["a", "b"]') == ["a", "b"]

    def test_embedded_array(self) -> None:
        text = 'Here are the facts: ["a", "b"] and that is all.'
        assert LLMMemoryExtractor._parse_json_array(text) == ["a", "b"]

    def test_code_fence(self) -> None:
        text = '```json\n["x"]\n```'
        assert LLMMemoryExtractor._parse_json_array(text) == ["x"]

    def test_empty(self) -> None:
        assert LLMMemoryExtractor._parse_json_array("") == []
        assert LLMMemoryExtractor._parse_json_array("no array here") == []

    def test_non_string_items(self) -> None:
        text = '[1, true, "hello", null]'
        result = LLMMemoryExtractor._parse_json_array(text)
        assert result == ["1", "True", "hello"]  # null filtered out


# ---------------------------------------------------------------------------
# Inheritance
# ---------------------------------------------------------------------------


class TestInheritance:
    """LLMMemoryExtractor should be a subclass of MemoryExtractor."""

    def test_is_subclass(self) -> None:
        assert issubclass(LLMMemoryExtractor, MemoryExtractor)

    def test_is_instance(self) -> None:
        llm = FakeLLM(["yes"])
        extractor = LLMMemoryExtractor(llm)
        assert isinstance(extractor, MemoryExtractor)
