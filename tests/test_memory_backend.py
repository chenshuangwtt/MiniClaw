"""Tests for memory/base.py and memory/extractor.py."""

import pytest
from abc import ABC

from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor


# ============================================================
# MemoryBackend abstract class
# ============================================================


class TestMemoryBackend:
    def test_is_abstract(self):
        assert issubclass(MemoryBackend, ABC)

    def test_cannot_instantiate(self):
        with pytest.raises(TypeError):
            MemoryBackend()  # type: ignore[abstract]

    def test_has_add_method(self):
        assert hasattr(MemoryBackend, "add")

    def test_has_search_method(self):
        assert hasattr(MemoryBackend, "search")


# ============================================================
# NullMemoryBackend
# ============================================================


class TestNullMemoryBackend:
    def test_is_subclass(self):
        assert issubclass(NullMemoryBackend, MemoryBackend)

    def test_instantiate(self):
        backend = NullMemoryBackend()
        assert backend is not None

    def test_add_does_nothing(self):
        backend = NullMemoryBackend()
        backend.add("remember this", "user1")
        backend.add("also this", "user1", {"source": "chat"})
        # Should not raise

    def test_search_returns_empty(self):
        backend = NullMemoryBackend()
        result = backend.search("anything", "user1")
        assert result == []

    def test_search_with_limit(self):
        backend = NullMemoryBackend()
        result = backend.search("query", "user1", limit=10)
        assert result == []

    def test_search_different_users(self):
        backend = NullMemoryBackend()
        assert backend.search("q", "alice") == []
        assert backend.search("q", "bob") == []


# ============================================================
# MemoryExtractor
# ============================================================


class TestMemoryExtractorShouldRemember:
    def setup_method(self):
        self.extractor = MemoryExtractor()

    def test_keyword_jizhu(self):
        assert self.extractor.should_remember("请记住我喜欢喝咖啡") is True

    def test_keyword_yihou(self):
        assert self.extractor.should_remember("以后不要这样做") is True

    def test_keyword_pianhao(self):
        assert self.extractor.should_remember("我的偏好是简洁风格") is True

    def test_keyword_wo_xihuan(self):
        assert self.extractor.should_remember("我喜欢用 Python") is True

    def test_keyword_wo_xiwang(self):
        assert self.extractor.should_remember("我希望代码有类型标注") is True

    def test_keyword_cong_xianzai(self):
        assert self.extractor.should_remember("从现在开始用中文回复") is True

    def test_no_keyword(self):
        assert self.extractor.should_remember("今天天气不错") is False

    def test_empty_string(self):
        assert self.extractor.should_remember("") is False

    def test_multiple_keywords(self):
        assert self.extractor.should_remember("记住，以后我喜欢用 vim") is True

    def test_partial_match_not_triggered(self):
        """'记忆' does not contain '记住' as a substring."""
        assert self.extractor.should_remember("这段记忆很重要") is False


class TestMemoryExtractorExtract:
    def setup_method(self):
        self.extractor = MemoryExtractor()

    def test_extract_single_sentence(self):
        result = self.extractor.extract("请记住我喜欢喝咖啡。")
        assert len(result) >= 1
        assert any("记住" in m for m in result)

    def test_extract_multiple_sentences(self):
        text = "今天天气不错。请记住我喜欢 Python。以后多写测试。"
        result = self.extractor.extract(text)
        assert len(result) >= 2
        assert any("记住" in m for m in result)
        assert any("以后" in m for m in result)

    def test_extract_no_keyword(self):
        result = self.extractor.extract("今天天气不错，适合出门。")
        assert result == []

    def test_extract_empty_string(self):
        result = self.extractor.extract("")
        assert result == []

    def test_extract_whitespace_only(self):
        result = self.extractor.extract("   ")
        assert result == []

    def test_extract_preserves_content(self):
        result = self.extractor.extract("我希望代码有类型标注。")
        assert len(result) >= 1
        assert "类型标注" in result[0]

    def test_extract_with_comma_split(self):
        """If no sentence-ending punctuation, split by comma."""
        result = self.extractor.extract("我喜欢 Python，也喜欢 Rust")
        assert len(result) >= 1


class TestMemoryExtractorCustomKeywords:
    def test_custom_keywords(self):
        extractor = MemoryExtractor(keywords=["重要", "注意"])
        assert extractor.should_remember("这个很重要") is True
        assert extractor.should_remember("请注意安全") is True
        assert extractor.should_remember("请记住这个") is False

    def test_custom_extract(self):
        extractor = MemoryExtractor(keywords=["重要"])
        result = extractor.extract("这个很重要。其他不重要。")
        assert len(result) >= 1
        assert any("重要" in m for m in result)
