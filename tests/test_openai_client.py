"""Tests for llm/openai_client.py.

These tests verify error handling and configuration without calling
the real OpenAI API.
"""

import os
import pytest
from unittest.mock import patch, MagicMock

from miniclaw.llm.openai_client import OpenAIClient
from miniclaw.llm.base import BaseLLM, LLMResponse


# Helper: patch the OpenAI class so no real HTTP/SSL is initialized
def _mock_openai_init():
    """Return a context manager that mocks 'from openai import OpenAI'."""
    return patch("miniclaw.llm.openai_client.OpenAI", new=MagicMock())


# ============================================================
# Initialization errors
# ============================================================


class TestInitErrors:
    def test_missing_api_key_raises(self):
        """No api_key param and no env var → clear error."""
        with patch.dict(os.environ, {}, clear=True):
            # Remove OPENAI_API_KEY if it exists
            os.environ.pop("OPENAI_API_KEY", None)
            with pytest.raises(ValueError, match="OPENAI_API_KEY"):
                OpenAIClient()

    def test_error_message_is_helpful(self):
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            try:
                OpenAIClient()
            except ValueError as exc:
                assert "environment variable" in str(exc).lower() or "api_key" in str(exc).lower()

    def test_explicit_api_key_works(self):
        """Passing api_key directly should not raise."""
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("OPENAI_API_KEY", None)
            with _mock_openai_init():
                client = OpenAIClient(api_key="sk-test-key-12345")
                assert client.model == "gpt-4o-mini"

    def test_env_var_api_key_works(self):
        """Reading from OPENAI_API_KEY env var should not raise."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert client.model == "gpt-4o-mini"

    def test_explicit_key_overrides_env(self):
        """Explicit key takes precedence over env var."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-key"}):
            with _mock_openai_init():
                client = OpenAIClient(api_key="sk-explicit-key")
                assert client.model == "gpt-4o-mini"


# ============================================================
# Configuration
# ============================================================


class TestConfiguration:
    def test_default_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert client.model == "gpt-4o-mini"

    def test_custom_model(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient(model="gpt-4o")
                assert client.model == "gpt-4o"

    def test_custom_base_url(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient(base_url="http://localhost:11434/v1")
                assert client.model == "gpt-4o-mini"

    def test_base_url_from_env(self):
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "sk-test",
                "OPENAI_BASE_URL": "http://localhost:11434/v1",
            },
        ):
            with _mock_openai_init():
                client = OpenAIClient()
                assert client.model == "gpt-4o-mini"


# ============================================================
# Interface compatibility
# ============================================================


class TestInterface:
    def test_is_subclass_of_base(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert isinstance(client, BaseLLM)

    def test_has_call_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert hasattr(client, "call")
                assert callable(client.call)

    def test_has_generate_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert hasattr(client, "generate")
                assert callable(client.generate)

    def test_has_chat_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert hasattr(client, "chat")
                assert callable(client.chat)

    def test_has_stream_method(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                assert hasattr(client, "stream")
                assert callable(client.stream)

    def test_generate_calls_call(self):
        """generate() should delegate to call()."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-test"}):
            with _mock_openai_init():
                client = OpenAIClient()
                mock_completion = MagicMock()
                mock_completion.choices = [MagicMock()]
                mock_completion.choices[0].message.content = "42"
                client._client.chat.completions.create = MagicMock(return_value=mock_completion)

                result = client.generate("What is 6*7?")
                assert result == "42"
                call_args = client._client.chat.completions.create.call_args
                messages = call_args.kwargs["messages"]
                assert len(messages) == 1
                assert messages[0]["role"] == "user"
                assert messages[0]["content"] == "What is 6*7?"


# ============================================================
# Mocked API calls
# ============================================================


class TestMockedCalls:
    def _make_client(self):
        with _mock_openai_init():
            return OpenAIClient(api_key="sk-test")

    def test_call_returns_content(self):
        client = self._make_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "Hello!"
        mock_completion.choices[0].message.tool_calls = None
        client._client.chat.completions.create = MagicMock(return_value=mock_completion)

        result = client.call([{"role": "user", "content": "Hi"}])
        assert result == "Hello!"

    def test_call_empty_content_returns_empty(self):
        client = self._make_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = None
        mock_completion.choices[0].message.tool_calls = None
        client._client.chat.completions.create = MagicMock(return_value=mock_completion)

        result = client.call([{"role": "user", "content": "Hi"}])
        assert result == ""

    def test_chat_returns_llm_response(self):
        client = self._make_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "ok"
        mock_completion.choices[0].message.tool_calls = None
        client._client.chat.completions.create = MagicMock(return_value=mock_completion)

        response = client.chat([{"role": "user", "content": "test"}])
        assert isinstance(response, LLMResponse)
        assert response.content == "ok"
        assert response.tool_calls == []

    def test_chat_with_tools(self):
        client = self._make_client()

        mock_tc = MagicMock()
        mock_tc.id = "call_123"
        mock_tc.function.name = "echo"
        mock_tc.function.arguments = '{"text": "hello"}'

        mock_message = MagicMock()
        mock_message.content = ""
        mock_message.tool_calls = [mock_tc]

        mock_choice = MagicMock()
        mock_choice.message = mock_message

        mock_completion = MagicMock()
        mock_completion.choices = [mock_choice]
        client._client.chat.completions.create = MagicMock(return_value=mock_completion)

        tools = [{"type": "function", "function": {"name": "echo"}}]
        response = client.chat(
            [{"role": "user", "content": "echo hello"}],
            tools=tools,
        )
        assert len(response.tool_calls) == 1
        assert response.tool_calls[0].name == "echo"
        assert response.tool_calls[0].arguments == {"text": "hello"}

    def test_chat_passes_temperature(self):
        client = self._make_client()
        mock_completion = MagicMock()
        mock_completion.choices = [MagicMock()]
        mock_completion.choices[0].message.content = "ok"
        mock_completion.choices[0].message.tool_calls = None
        client._client.chat.completions.create = MagicMock(return_value=mock_completion)

        client.chat([{"role": "user", "content": "test"}], temperature=0.7)
        call_args = client._client.chat.completions.create.call_args
        assert call_args.kwargs["temperature"] == 0.7

    def test_stream_yields_content_chunks(self):
        client = self._make_client()

        event1 = MagicMock()
        event1.choices = [MagicMock()]
        event1.choices[0].delta.content = "Hel"
        event2 = MagicMock()
        event2.choices = [MagicMock()]
        event2.choices[0].delta.content = "lo"
        event3 = MagicMock()
        event3.choices = [MagicMock()]
        event3.choices[0].delta.content = None

        client._client.chat.completions.create = MagicMock(return_value=[event1, event2, event3])

        chunks = list(client.stream([{"role": "user", "content": "Hi"}], temperature=0.2))

        assert chunks == ["Hel", "lo"]
        call_args = client._client.chat.completions.create.call_args
        assert call_args.kwargs["stream"] is True
        assert call_args.kwargs["temperature"] == 0.2

    def test_call_stream_delegates_to_stream(self):
        client = self._make_client()

        event = MagicMock()
        event.choices = [MagicMock()]
        event.choices[0].delta.content = "chunk"
        client._client.chat.completions.create = MagicMock(return_value=[event])

        assert list(client.call_stream([{"role": "user", "content": "Hi"}])) == ["chunk"]
