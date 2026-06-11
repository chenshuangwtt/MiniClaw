"""Tests for MCP tool adapter."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from miniclaw.tools.mcp_adapter import MCPClient, MCPToolAdapter, MCPToolRegistry, _convert_schema
from miniclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# _convert_schema
# ---------------------------------------------------------------------------


class TestConvertSchema:
    def test_input_schema_wrapper(self) -> None:
        mcp = {
            "name": "test",
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        result = _convert_schema(mcp)
        assert result["type"] == "object"
        assert "query" in result["properties"]

    def test_direct_schema(self) -> None:
        mcp = {
            "type": "object",
            "properties": {"x": {"type": "integer"}},
        }
        result = _convert_schema(mcp)
        assert result["type"] == "object"

    def test_non_dict_returns_empty(self) -> None:
        result = _convert_schema({"inputSchema": "invalid"})
        assert result["type"] == "object"
        assert result["properties"] == {}

    def test_missing_type_adds_object(self) -> None:
        result = _convert_schema({"properties": {"a": {"type": "string"}}})
        assert result["type"] == "object"


# ---------------------------------------------------------------------------
# MCPToolAdapter
# ---------------------------------------------------------------------------


class TestMCPToolAdapter:
    def test_basic_properties(self) -> None:
        client = MagicMock(spec=MCPClient)
        schema = {
            "inputSchema": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        }
        tool = MCPToolAdapter("search", schema, client, description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert "query" in tool.schema["properties"]

    def test_default_description(self) -> None:
        client = MagicMock(spec=MCPClient)
        schema = {"description": "A test tool", "inputSchema": {"type": "object"}}
        tool = MCPToolAdapter("my_tool", schema, client)
        assert tool.description == "A test tool"

    def test_run_calls_client(self) -> None:
        client = MagicMock(spec=MCPClient)
        client.call_tool.return_value = "result"
        schema = {"inputSchema": {"type": "object", "properties": {"q": {"type": "string"}}}}
        tool = MCPToolAdapter("search", schema, client)

        result = tool.run(q="hello")
        client.call_tool.assert_called_once_with("search", {"q": "hello"})
        assert result == "result"


# ---------------------------------------------------------------------------
# MCPClient (mocked subprocess)
# ---------------------------------------------------------------------------


class TestMCPClient:
    def _make_mock_process(self, responses: list[str]) -> MagicMock:
        """Create a mock subprocess with pre-loaded responses."""
        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdin = MagicMock()
        mock_stdout.readline.side_effect = responses
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        return mock_process

    def test_list_tools(self) -> None:
        """Test list_tools with a mock subprocess."""
        client = MCPClient("fake-server")

        # Only need one response since we bypass start()
        responses = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {
                        "tools": [
                            {
                                "name": "search",
                                "description": "Search",
                                "inputSchema": {"type": "object"},
                            },
                            {
                                "name": "fetch",
                                "description": "Fetch URL",
                                "inputSchema": {"type": "object"},
                            },
                        ]
                    },
                }
            )
            + "\n",
        ]
        client._process = self._make_mock_process(responses)

        tools = client.list_tools()
        assert len(tools) == 2
        assert tools[0]["name"] == "search"
        assert tools[1]["name"] == "fetch"

    def test_call_tool(self) -> None:
        client = MCPClient("fake-server")

        responses = [
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "result": {"content": [{"type": "text", "text": "search results"}]},
                }
            )
            + "\n",
        ]
        client._process = self._make_mock_process(responses)

        result = client.call_tool("search", {"query": "test"})
        assert result == "search results"

    def test_request_timeout_stops_process(self) -> None:
        client = MCPClient("fake-server", timeout=0.1)

        mock_process = MagicMock()
        mock_stdout = MagicMock()
        mock_stdin = MagicMock()

        def slow_readline() -> str:
            time.sleep(1)
            return ""

        mock_stdout.readline.side_effect = slow_readline
        mock_process.stdin = mock_stdin
        mock_process.stdout = mock_stdout
        client._process = mock_process

        start = time.perf_counter()
        with pytest.raises(TimeoutError, match="timed out"):
            client.list_tools()

        assert time.perf_counter() - start < 0.75
        mock_process.terminate.assert_called_once()

    def test_context_manager(self) -> None:
        """Test start/stop lifecycle."""
        client = MCPClient("echo", args=["test"])
        # We can't actually start a real server, but test the guard
        with pytest.raises(RuntimeError, match="not started"):
            client._send_request("test", {})


# ---------------------------------------------------------------------------
# MCPToolRegistry
# ---------------------------------------------------------------------------


class TestMCPToolRegistry:
    def test_discover_tools(self) -> None:
        """Test tool discovery with mocked client."""
        mock_client = MagicMock(spec=MCPClient)
        mock_client.list_tools.return_value = [
            {"name": "tool_a", "description": "Tool A", "inputSchema": {"type": "object"}},
            {
                "name": "tool_b",
                "description": "Tool B",
                "inputSchema": {"type": "object", "properties": {"x": {"type": "integer"}}},
            },
        ]

        with patch("miniclaw.tools.mcp_adapter.MCPClient", return_value=mock_client):
            registry = MCPToolRegistry({"command": "fake-server"})
            tools = registry.discover_tools()

        assert len(tools) == 2
        assert tools[0].name == "tool_a"
        assert tools[1].name == "tool_b"

    def test_register_all(self) -> None:
        """Test registering tools with a ToolRegistry."""
        mock_client = MagicMock(spec=MCPClient)
        mock_client.list_tools.return_value = [
            {"name": "mcp_tool", "description": "An MCP tool", "inputSchema": {"type": "object"}},
        ]

        with patch("miniclaw.tools.mcp_adapter.MCPClient", return_value=mock_client):
            mcp_registry = MCPToolRegistry({"command": "fake-server"})
            tool_registry = ToolRegistry()
            count = mcp_registry.register_all(tool_registry)

        assert count == 1
        assert tool_registry.get("mcp_tool") is not None

    def test_missing_command_raises(self) -> None:
        registry = MCPToolRegistry({})
        with pytest.raises(ValueError, match="command"):
            registry.discover_tools()
