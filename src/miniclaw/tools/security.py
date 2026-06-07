"""Small safety helpers for tool implementations."""

from __future__ import annotations

from pathlib import Path


def resolve_workspace_path(
    path: str, workspace_root: str | Path | None = None
) -> tuple[Path | None, str | None]:
    """Resolve *path* and optionally enforce that it stays inside *workspace_root*.

    Args:
        path: User-provided path.
        workspace_root: Optional workspace boundary. When omitted, the path is
            only expanded/resolved.

    Returns:
        ``(resolved_path, None)`` on success, or ``(None, error_message)``.
    """
    try:
        target = Path(path).expanduser().resolve()
    except OSError as exc:
        return None, f"Invalid path {path}: {exc}"

    if workspace_root is None:
        return target, None

    try:
        root = Path(workspace_root).expanduser().resolve()
    except OSError as exc:
        return None, f"Invalid workspace root {workspace_root}: {exc}"

    if target == root or root in target.parents:
        return target, None

    return None, f"Path is outside workspace: {path}"
