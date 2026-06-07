"""Tests for agent/config.py."""

from pathlib import Path

from miniclaw.agent.config import (
    AgentConfig,
    Config,
    LLMConfig,
    StorageConfig,
    ToolsConfig,
    load_config,
)


# ============================================================
# Data classes
# ============================================================


class TestLLMConfig:
    def test_defaults(self):
        c = LLMConfig()
        assert c.provider == "fake"
        assert c.model == "gpt-4o-mini"
        assert c.api_key is None
        assert c.base_url is None


class TestAgentConfig:
    def test_defaults(self):
        c = AgentConfig()
        assert c.max_steps == 10
        assert c.max_errors == 3
        assert c.system_prompt == ""


class TestStorageConfig:
    def test_defaults(self):
        c = StorageConfig()
        assert c.db_path == ".miniclaw/miniclaw.db"


class TestToolsConfig:
    def test_defaults(self):
        c = ToolsConfig()
        assert c.allow_file_write is True
        assert c.allow_shell is True
        assert c.allow_search is False
        assert c.shell_allowed_prefixes is None


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.llm.provider == "fake"
        assert c.agent.max_steps == 10
        assert c.storage.db_path == ".miniclaw/miniclaw.db"
        assert c.tools.allow_shell is True

    def test_override_simple(self):
        c = Config()
        c.override(**{"llm.provider": "openai"})
        assert c.llm.provider == "openai"

    def test_override_none_ignored(self):
        c = Config()
        c.override(**{"llm.provider": None})
        assert c.llm.provider == "fake"

    def test_override_nested(self):
        c = Config()
        c.override(**{"llm.model": "gpt-4o"})
        assert c.llm.model == "gpt-4o"


# ============================================================
# Loader
# ============================================================


class TestLoadConfig:
    def test_no_file_returns_defaults(self):
        config = load_config("/nonexistent/path.toml")
        assert config.llm.provider == "fake"
        assert config.agent.max_steps == 10

    def test_none_returns_defaults(self):
        config = load_config(None)
        assert config.llm.provider == "fake"

    def test_load_from_file(self, tmp_path: Path):
        f = tmp_path / "test.toml"
        f.write_text("""
[llm]
provider = "openai"
model = "gpt-4o"

[agent]
max_steps = 20
max_errors = 5

[storage]
db_path = "test.db"

[tools]
allow_file_write = false
allow_shell = false
allow_search = true
shell_allowed_prefixes = ["echo", "python -m pytest"]
""")
        config = load_config(f)
        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.agent.max_steps == 20
        assert config.agent.max_errors == 5
        assert config.storage.db_path == "test.db"
        assert config.tools.allow_file_write is False
        assert config.tools.allow_shell is False
        assert config.tools.allow_search is True
        assert config.tools.shell_allowed_prefixes == ["echo", "python -m pytest"]

    def test_partial_config(self, tmp_path: Path):
        f = tmp_path / "partial.toml"
        f.write_text("""
[llm]
model = "gpt-4o"
""")
        config = load_config(f)
        assert config.llm.model == "gpt-4o"
        assert config.llm.provider == "fake"  # default
        assert config.agent.max_steps == 10  # default

    def test_empty_file(self, tmp_path: Path):
        f = tmp_path / "empty.toml"
        f.write_text("")
        config = load_config(f)
        assert config.llm.provider == "fake"

    def test_invalid_toml_returns_defaults(self, tmp_path: Path):
        f = tmp_path / "bad.toml"
        f.write_text("this is not valid [[[ toml")
        config = load_config(f)
        assert config.llm.provider == "fake"

    def test_unknown_keys_ignored(self, tmp_path: Path):
        f = tmp_path / "extra.toml"
        f.write_text("""
[llm]
provider = "fake"
unknown_key = "value"
""")
        config = load_config(f)
        assert config.llm.provider == "fake"
        assert not hasattr(config.llm, "unknown_key")

    def test_env_overrides_without_config_file(self, monkeypatch):
        monkeypatch.setenv("MINICLAW_LLM_PROVIDER", "openai")
        monkeypatch.setenv("MINICLAW_MODEL", "gpt-4o")
        monkeypatch.setenv("MINICLAW_MAX_STEPS", "7")
        monkeypatch.setenv("MINICLAW_DB_PATH", ".miniclaw/env.db")
        monkeypatch.setenv("MINICLAW_ALLOW_SHELL", "false")
        monkeypatch.setenv("MINICLAW_ALLOW_SEARCH", "true")
        monkeypatch.setenv("MINICLAW_SHELL_ALLOWED_PREFIXES", "echo,python")

        config = load_config("/nonexistent/path.toml")

        assert config.llm.provider == "openai"
        assert config.llm.model == "gpt-4o"
        assert config.agent.max_steps == 7
        assert config.storage.db_path == ".miniclaw/env.db"
        assert config.tools.allow_shell is False
        assert config.tools.allow_search is True
        assert config.tools.shell_allowed_prefixes == ["echo", "python"]

    def test_invalid_int_env_is_ignored(self, monkeypatch):
        monkeypatch.setenv("MINICLAW_MAX_STEPS", "not-an-int")
        config = load_config("/nonexistent/path.toml")
        assert config.agent.max_steps == 10
