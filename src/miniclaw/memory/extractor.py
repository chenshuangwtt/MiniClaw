"""Memory extractor — decides whether text is worth remembering, and what to extract.

Features:
    - Keyword-based heuristics (configurable).
    - Sensitive information filtering (API keys, passwords, etc.).
    - LLM-based extractor for semantic memory extraction.

Two extractor classes are available:

    - ``MemoryExtractor`` — keyword/regex-based, zero dependencies.
    - ``LLMMemoryExtractor`` — uses an LLM for semantic judgment and extraction.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

# Keywords that indicate the user wants something remembered.
_REMEMBER_KEYWORDS: list[str] = [
    "记住",
    "以后",
    "偏好",
    "我喜欢",
    "我希望",
    "从现在开始",
]

# Patterns that indicate sensitive information — must NOT be stored.
_SENSITIVE_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"sk-[a-zA-Z0-9]{20,}"),  # OpenAI-style API keys
    re.compile(r"password\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"secret\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"token\s*[:=]\s*\S+", re.IGNORECASE),
    re.compile(r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b"),  # credit card
    re.compile(r"-----BEGIN.*PRIVATE KEY-----"),  # PEM private keys
]


def contains_sensitive(text: str) -> bool:
    """Check whether *text* contains sensitive information (API keys, passwords, etc.)."""
    return any(p.search(text) for p in _SENSITIVE_PATTERNS)


class MemoryExtractor:
    """Determines whether a message should be stored as a memory,
    and extracts the relevant fragments.

    Usage::

        extractor = MemoryExtractor()

        if extractor.should_remember(user_text):
            memories = extractor.extract(user_text)
            for m in memories:
                backend.add(m, user_id)
    """

    def __init__(self, keywords: list[str] | None = None) -> None:
        """Initialize with optional custom keywords.

        Args:
            keywords: Override the default keyword list.
        """
        self._keywords = keywords if keywords is not None else _REMEMBER_KEYWORDS

    def should_remember(self, text: str) -> bool:
        """Check whether *text* contains any memory-worthy keywords.

        Returns ``False`` if the text contains sensitive information.

        Args:
            text: The message text to check.

        Returns:
            ``True`` if at least one keyword is found and no sensitive data.
        """
        if contains_sensitive(text):
            return False
        return any(kw in text for kw in self._keywords)

    def extract(self, text: str) -> list[str]:
        """Extract memory-worthy fragments from *text*.

        Filters out sentences containing sensitive information.

        Args:
            text: The message text to extract from.

        Returns:
            List of extracted memory strings (may be empty).
        """
        if not text or not text.strip():
            return []

        memories: list[str] = []
        sentences = _split_sentences(text)

        for sentence in sentences:
            if any(kw in sentence for kw in self._keywords):
                cleaned = sentence.strip()
                if cleaned and not contains_sensitive(cleaned):
                    memories.append(cleaned)

        return memories


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences by Chinese and English punctuation."""
    parts = re.split(r"(?<=[。！？.!?\n])\s*", text)
    if len(parts) <= 1:
        parts = re.split(r"[，,；;]", text)
    return [p.strip() for p in parts if p.strip()]


# ---------------------------------------------------------------------------
# LLM-based extractor
# ---------------------------------------------------------------------------

_SHOULD_REMEMBER_PROMPT = """\
You are a memory filter. Decide whether the following text contains information
worth storing as long-term memory (user preferences, facts, instructions, etc.).

Reply with ONLY "yes" or "no".

Text:
{text}"""

_EXTRACT_PROMPT = """\
You are a memory extractor. Extract individual facts or preferences worth
remembering from the following text. Return a JSON array of strings.
If nothing is worth remembering, return an empty array [].

Examples of memorable information:
- User preferences ("I prefer dark mode")
- Personal facts ("My name is Alice")
- Instructions ("Always use metric units")
- Important context ("The production server is at 10.0.0.1")

Do NOT include:
- Sensitive information (passwords, API keys, tokens)
- Transient requests ("What's the weather?")
- Generic conversation

Text:
{text}

Return ONLY a JSON array, e.g. ["fact 1", "fact 2"]."""


class LLMMemoryExtractor(MemoryExtractor):
    """Memory extractor powered by an LLM for semantic understanding.

    Uses the LLM to judge whether text is worth remembering and to extract
    individual memory fragments. Falls back to keyword-based extraction if
    the LLM call fails.

    Args:
        llm: A ``BaseLLM`` instance for making extraction calls.
        keywords: Optional override for the keyword fallback list.
        fallback_on_error: If ``True`` (default), fall back to keyword-based
            extraction when the LLM call fails.

    Usage::

        from miniclaw.llm.openai_client import OpenAIClient
        from miniclaw.memory.extractor import LLMMemoryExtractor

        llm = OpenAIClient(model="gpt-4o-mini")
        extractor = LLMMemoryExtractor(llm)

        if extractor.should_remember("I prefer dark mode"):
            memories = extractor.extract("I prefer dark mode")
    """

    def __init__(
        self,
        llm: Any,
        keywords: list[str] | None = None,
        fallback_on_error: bool = True,
    ) -> None:
        super().__init__(keywords=keywords)
        self._llm = llm
        self._fallback_on_error = fallback_on_error

    def should_remember(self, text: str) -> bool:
        """Use the LLM to decide whether *text* is worth remembering.

        Falls back to keyword-based check if the LLM call fails and
        ``fallback_on_error`` is ``True``.
        """
        if contains_sensitive(text):
            return False

        try:
            prompt = _SHOULD_REMEMBER_PROMPT.format(text=text)
            response = self._llm.generate(prompt).strip().lower()
            return response.startswith("yes")
        except Exception as exc:
            logger.warning("LLM should_remember failed: %s", exc)
            if self._fallback_on_error:
                return super().should_remember(text)
            return False

    def extract(self, text: str) -> list[str]:
        """Use the LLM to extract memory-worthy fragments from *text*.

        Falls back to keyword-based extraction if the LLM call fails and
        ``fallback_on_error`` is ``True``.
        """
        if not text or not text.strip():
            return []

        if contains_sensitive(text):
            return []

        try:
            prompt = _EXTRACT_PROMPT.format(text=text)
            response = self._llm.generate(prompt).strip()
            memories = self._parse_json_array(response)
            # Filter out any sensitive fragments the LLM might have included
            return [m for m in memories if m and not contains_sensitive(m)]
        except Exception as exc:
            logger.warning("LLM extract failed: %s", exc)
            if self._fallback_on_error:
                return super().extract(text)
            return []

    @staticmethod
    def _parse_json_array(text: str) -> list[str]:
        """Parse a JSON array from LLM output, handling markdown fences."""
        # Strip markdown code fences if present
        text = text.strip()
        fence_match = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        # Try direct parse
        try:
            result = json.loads(text)
            if isinstance(result, list):
                return [str(item) for item in result if item]
        except json.JSONDecodeError:
            pass

        # Try finding the first [...] in the text
        bracket_match = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket_match:
            try:
                result = json.loads(bracket_match.group())
                if isinstance(result, list):
                    return [str(item) for item in result if item]
            except json.JSONDecodeError:
                pass

        return []
