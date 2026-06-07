"""File system tools: list_files, read_file, write_file."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from miniclaw.tools.base import Tool
from miniclaw.tools.security import resolve_workspace_path

# Maximum characters to return from a single file read.
MAX_READ_CHARS = 50_000


class ListFiles(Tool):
    """List files and directories at a given path."""

    name = "list_files"
    description = (
        "List files and directories in the given path. "
        "Returns a list of entries with their type (file/directory). "
        "Does not recurse into subdirectories."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Directory path to list.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = workspace_root

    def run(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """List entries in *path*.

        Returns:
            A dict with ``entries`` (list of {name, type}) or ``error``.
        """
        target, error = resolve_workspace_path(path, self.workspace_root)
        if error:
            return {"error": error}
        assert target is not None

        if not target.exists():
            return {"error": f"Path does not exist: {path}"}
        if not target.is_dir():
            return {"error": f"Not a directory: {path}"}

        entries: list[dict[str, str]] = []
        try:
            for item in sorted(target.iterdir()):
                entries.append(
                    {
                        "name": item.name,
                        "type": "directory" if item.is_dir() else "file",
                    }
                )
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

        return {"path": str(target.resolve()), "entries": entries}


class ReadFile(Tool):
    """Read the contents of a file."""

    name = "read_file"
    description = (
        "Read and return the content of a file. "
        f"Content is truncated at {MAX_READ_CHARS} characters."
    )
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to read.",
            },
        },
        "required": ["path"],
    }

    def __init__(self, workspace_root: str | Path | None = None) -> None:
        self.workspace_root = workspace_root

    def run(self, path: str, **kwargs: Any) -> dict[str, Any]:
        """Read file at *path*.

        Returns:
            A dict with ``content``, ``truncated``, ``chars`` or ``error``.
        """
        target, error = resolve_workspace_path(path, self.workspace_root)
        if error:
            return {"error": error}
        assert target is not None

        if not target.exists():
            return {"error": f"Path does not exist: {path}"}
        if not target.is_file():
            return {"error": f"Not a file: {path}"}

        try:
            raw = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            return {"error": f"File is not valid UTF-8: {path}"}
        except PermissionError:
            return {"error": f"Permission denied: {path}"}

        truncated = len(raw) > MAX_READ_CHARS
        content = raw[:MAX_READ_CHARS]

        return {
            "path": str(target.resolve()),
            "content": content,
            "truncated": truncated,
            "chars": len(raw),
        }


class WriteFile(Tool):
    """Write content to a file."""

    name = "write_file"
    description = "Write content to a file. Creates parent directories if they don't exist."
    schema = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "File path to write.",
            },
            "content": {
                "type": "string",
                "description": "Content to write into the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, workspace_root: str | Path | None = None, allow_write: bool = True) -> None:
        self.workspace_root = workspace_root
        self.allow_write = allow_write

    def run(self, path: str, content: str, **kwargs: Any) -> dict[str, Any]:
        """Write *content* to *path*.

        Returns:
            A dict with ``path``, ``chars_written`` or ``error``.
        """
        if not self.allow_write:
            return {"error": "File writes are disabled by tool permissions."}

        target, error = resolve_workspace_path(path, self.workspace_root)
        if error:
            return {"error": error}
        assert target is not None

        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(content, encoding="utf-8")
        except PermissionError:
            return {"error": f"Permission denied: {path}"}
        except OSError as exc:
            return {"error": f"OS error writing {path}: {exc}"}

        return {
            "path": str(target.resolve()),
            "chars_written": len(content),
        }
