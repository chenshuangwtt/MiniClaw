"""Memory extractor — decides whether text is worth remembering, and what to extract.

Uses keyword-based heuristics.  Can be replaced with an LLM-based
extractor later without changing the interface.
"""

from __future__ import annotations

# Keywords that indicate the user wants something remembered.
_REMEMBER_KEYWORDS: list[str] = [
    "记住",
    "以后",
    "偏好",
    "我喜欢",
    "我希望",
    "从现在开始",
]


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

        Args:
            text: The message text to check.

        Returns:
            ``True`` if at least one keyword is found.
        """
        return any(kw in text for kw in self._keywords)

    def extract(self, text: str) -> list[str]:
        """Extract memory-worthy fragments from *text*.

        Strategy: find each keyword occurrence and extract the
        sentence or clause containing it.

        Args:
            text: The message text to extract from.

        Returns:
            List of extracted memory strings (may be empty).
        """
        if not text or not text.strip():
            return []

        memories: list[str] = []
        # Split by common sentence delimiters
        sentences = _split_sentences(text)

        for sentence in sentences:
            if any(kw in sentence for kw in self._keywords):
                cleaned = sentence.strip()
                if cleaned:
                    memories.append(cleaned)

        return memories


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences by Chinese and English punctuation."""
    import re

    # Split on Chinese/English sentence-ending punctuation
    parts = re.split(r"(?<=[。！？.!?\n])\s*", text)
    # Also split on commas if no sentence-ending punctuation was found
    if len(parts) <= 1:
        parts = re.split(r"[，,；;]", text)
    return [p.strip() for p in parts if p.strip()]
