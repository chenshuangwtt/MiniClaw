"""Memory extractor — decides whether text is worth remembering, and what to extract.

Features:
    - Keyword-based heuristics (configurable).
    - Sensitive information filtering (API keys, passwords, etc.).
    - Can be replaced with an LLM-based extractor later.
"""

from __future__ import annotations

import re

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
