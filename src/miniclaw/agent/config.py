"""Configuration loader for MiniClaw.

Reads settings from a TOML config file and merges them with CLI arguments.
CLI arguments always take precedence over config file values.

Config file search order (first found wins):
    1. Explicit path via ``--config``
    2. ``./miniclaw.toml`` (project root)
    3. ``~/.miniclaw/config.toml`` (user home)

Usage::

    from miniclaw.agent.config import load_config

    # Load with defaults
    config = load_config()

    # Load from explicit path
    config = load_config("/path/to/miniclaw.toml")

    # Access values
    config.llm.provider        # "fake" | "openai"
    config.llm.model           # "gpt-4o-mini"
    config.llm.api_key         # "sk-..." or None
    config.llm.base_url        # "http://..." or None
    config.agent.max_steps     # 10
    config.agent.max_errors    # 3
    config.storage.db_path     # ".miniclaw/miniclaw.db"
    config.tools.allow_search  # False
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Python 3.11+ has tomllib in stdlib
try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[no-redef]


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass
class LLMConfig:
    """LLM backend configuration."""

    provider: str = "fake"
    model: str = "gpt-4o-mini"
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class AgentConfig:
    """Agent loop configuration."""

    max_steps: int = 10
    max_errors: int = 3
    system_prompt: str = ""


@dataclass
class StorageConfig:
    """Storage configuration."""

    db_path: str = ".miniclaw/miniclaw.db"


@dataclass
class ToolsConfig:
    """Built-in tool permission configuration."""

    allow_file_write: bool = False  # Off by default — explicit opt-in required
    allow_shell: bool = False  # Off by default — explicit opt-in required
    allow_search: bool = False  # Off by default — explicit opt-in required
    shell_allowed_prefixes: list[str] | None = None


@dataclass
class Config:
    """Top-level configuration container."""

    llm: LLMConfig = field(default_factory=LLMConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    storage: StorageConfig = field(default_factory=StorageConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)

    def override(self, **kwargs: Any) -> None:
        """Override config values from CLI arguments.

        Only overrides values that are not ``None``.
        Supports dotted keys like ``llm.model``.
        """
        for key, value in kwargs.items():
            if value is None:
                continue
            parts = key.split(".")
            obj = self
            for part in parts[:-1]:
                obj = getattr(obj, part)
            setattr(obj, parts[-1], value)


# ------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------

# Default search paths
_SEARCH_PATHS = [
    Path("miniclaw.toml"),
    Path.home() / ".miniclaw" / "config.toml",
]


def load_config(path: str | Path | None = None) -> Config:
    """Load configuration from a TOML file.

    Args:
        path: Explicit path to config file.  If ``None``, searches
            default locations.

    Returns:
        A ``Config`` instance with values from the file (or defaults).
    """
    config = Config()

    # Find config file
    config_path = _find_config(path)
    if config_path is None:
        _apply_env_overrides(config)
        return config

    # Parse TOML
    try:
        with open(config_path, "rb") as f:
            data = tomllib.load(f)
    except Exception:
        _apply_env_overrides(config)
        return config

    # Apply to config
    _apply_section(config.llm, data.get("llm", {}))
    _apply_section(config.agent, data.get("agent", {}))
    _apply_section(config.storage, data.get("storage", {}))
    _apply_section(config.tools, data.get("tools", {}))
    _apply_env_overrides(config)

    return config


def _find_config(explicit: str | Path | None = None) -> Path | None:
    """Find the config file to use."""
    if explicit:
        p = Path(explicit)
        return p if p.exists() else None

    for p in _SEARCH_PATHS:
        if p.exists():
            return p

    return None


def _apply_section(obj: Any, data: dict[str, Any]) -> None:
    """Apply a dict of values to a dataclass instance."""
    for key, value in data.items():
        if hasattr(obj, key):
            setattr(obj, key, value)


def _apply_env_overrides(config: Config) -> None:
    """Apply MINICLAW_* environment variable overrides."""
    env_map: dict[str, tuple[str, str, type]] = {
        "MINICLAW_LLM_PROVIDER": ("llm", "provider", str),
        "MINICLAW_MODEL": ("llm", "model", str),
        "MINICLAW_API_KEY": ("llm", "api_key", str),
        "MINICLAW_BASE_URL": ("llm", "base_url", str),
        "MINICLAW_MAX_STEPS": ("agent", "max_steps", int),
        "MINICLAW_MAX_ERRORS": ("agent", "max_errors", int),
        "MINICLAW_DB_PATH": ("storage", "db_path", str),
        "MINICLAW_ALLOW_FILE_WRITE": ("tools", "allow_file_write", bool),
        "MINICLAW_ALLOW_SHELL": ("tools", "allow_shell", bool),
        "MINICLAW_ALLOW_SEARCH": ("tools", "allow_search", bool),
    }

    for env_name, (section, attr, caster) in env_map.items():
        raw = os.getenv(env_name)
        if raw is None:
            continue
        value = _coerce_env_value(raw, caster)
        if value is not None:
            setattr(getattr(config, section), attr, value)

    prefixes = os.getenv("MINICLAW_SHELL_ALLOWED_PREFIXES")
    if prefixes is not None:
        values = [p.strip() for p in prefixes.split(",") if p.strip()]
        config.tools.shell_allowed_prefixes = values or None


def _coerce_env_value(raw: str, caster: type) -> Any:
    """Best-effort environment value coercion."""
    if caster is bool:
        return raw.strip().lower() in {"1", "true", "yes", "on"}
    if caster is int:
        try:
            return int(raw)
        except ValueError:
            return None
    return raw
