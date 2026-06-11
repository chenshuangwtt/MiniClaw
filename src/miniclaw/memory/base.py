"""Memory backend abstraction.

Defines the interface for storing and retrieving memories,
plus a no-op implementation for testing or when memory is disabled.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class MemoryBackend(ABC):
    """Abstract base class for memory storage backends.

    A memory backend stores text fragments associated with a user,
    and supports keyword-based retrieval.

    Usage::

        class MyBackend(MemoryBackend):
            def add(self, text, user_id, metadata=None):
                ...  # store somewhere

            def search(self, query, user_id, limit=5):
                ...  # return matching texts
    """

    @abstractmethod
    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Store a memory entry.

        Args:
            text: The text to remember.
            user_id: The user this memory belongs to.
            metadata: Optional extra data (timestamp, source, etc.).
        """
        ...

    @abstractmethod
    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Search for memories matching *query*.

        Args:
            query: Search term.
            user_id: Restrict to this user's memories.
            limit: Maximum number of results.

        Returns:
            List of matching memory texts.
        """
        ...

    def remove(self, text: str, user_id: str) -> bool:
        """Remove a memory entry matching *text* for *user_id*.

        This is an optional method — backends that don't support removal
        can leave the default (returns ``False``).

        Args:
            text: The exact text to remove.
            user_id: User identifier.

        Returns:
            ``True`` if a memory was removed, ``False`` otherwise.
        """
        return False


class NullMemoryBackend(MemoryBackend):
    """No-op memory backend — does nothing, returns nothing.

    Useful as a default when memory is disabled, or as a placeholder
    in tests that don't care about memory behavior.
    """

    def add(
        self,
        text: str,
        user_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        """Discard the memory."""
        pass

    def search(
        self,
        query: str,
        user_id: str,
        limit: int = 5,
    ) -> list[str]:
        """Always return an empty list."""
        return []
