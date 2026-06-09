"""Memory abstraction layer."""

from miniclaw.memory.base import MemoryBackend, NullMemoryBackend
from miniclaw.memory.extractor import MemoryExtractor, contains_sensitive
from miniclaw.memory.manager import MemoryManager
from miniclaw.memory.mem0_store import Mem0MemoryBackend
from miniclaw.memory.vector import VectorMemoryBackend

__all__ = [
    "MemoryBackend",
    "NullMemoryBackend",
    "MemoryExtractor",
    "MemoryManager",
    "Mem0MemoryBackend",
    "VectorMemoryBackend",
    "contains_sensitive",
]
