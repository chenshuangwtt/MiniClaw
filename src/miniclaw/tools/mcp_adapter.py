"""MCP (Model Context Protocol) tool adapter.

Wraps MCP tools as MiniClaw ``Tool`` instances so they can be used
in the agent loop alongside built-in tools.

Supports two transport modes:
    - **stdio**: Spawns an MCP server as a subprocess and communicates
      via JSON-RPC 2.0 over stdin/stdout.
    - **SSE**: Connects to an MCP server over HTTP SSE (not yet implemented).

Usage::

    from miniclaw.tools.mcp_adapter import MCPToolRegistry

    registry = MCPToolRegistry({"command": "mcp-server", "args": ["--port", "3000"]})
    tools = registry.discover_tools()
    for tool in tools:
        agent_registry.register(tool)
"""

from __future__ import annotations

import json
import logging
import queue
import subprocess
import threading
from typing import Any

from miniclaw.tools.base import Tool

logger = logging.getLogger(__name__)


class MCPClient:
    """Minimal MCP client using JSON-RPC 2.0 over stdio.

    Spawns a subprocess, sends JSON-RPC requests, and reads responses.

    Args:
        command: The command to start the MCP server.
        args: Arguments for the command.
        timeout: Default timeout for requests in seconds.
    """

    def __init__(
        self,
        command: str,
        args: list[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._command = command
        self._args = args or []
        self._timeout = timeout
        self._process: subprocess.Popen | None = None
        self._request_id = 0
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the MCP server subprocess."""
        if self._process is not None:
            return

        cmd = [self._command] + self._args
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        # Initialize the connection
        self._send_request(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "miniclaw", "version": "0.4.0"},
            },
        )
        # Send initialized notification
        self._send_notification("notifications/initialized", {})

    def stop(self) -> None:
        """Stop the MCP server subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
            self._process = None

    def list_tools(self) -> list[dict[str, Any]]:
        """Request the list of available tools from the server.

        Returns:
            List of tool definitions from the MCP server.
        """
        result = self._send_request("tools/list", {})
        return result.get("tools", [])

    def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool on the MCP server.

        Args:
            name: Tool name.
            arguments: Tool arguments.

        Returns:
            The tool result.
        """
        result = self._send_request(
            "tools/call",
            {
                "name": name,
                "arguments": arguments,
            },
        )
        # MCP returns content as a list of content blocks
        content = result.get("content", [])
        if isinstance(content, list) and len(content) == 1:
            block = content[0]
            if isinstance(block, dict) and block.get("type") == "text":
                return block.get("text", "")
        return content

    def _send_request(self, method: str, params: dict[str, Any]) -> dict[str, Any]:
        """Send a JSON-RPC 2.0 request and wait for the response."""
        with self._lock:
            self._request_id += 1
            request = {
                "jsonrpc": "2.0",
                "id": self._request_id,
                "method": method,
                "params": params,
            }

            if self._process is None or self._process.stdin is None or self._process.stdout is None:
                raise RuntimeError("MCP client not started")

            # Send request
            payload = json.dumps(request) + "\n"
            self._process.stdin.write(payload)
            self._process.stdin.flush()

            # Read response
            response_line = self._read_response_line()
            if not response_line:
                raise RuntimeError("MCP server closed connection")

            response = json.loads(response_line)

            if "error" in response:
                error = response["error"]
                raise RuntimeError(f"MCP error {error.get('code')}: {error.get('message')}")

            return response.get("result", {})

    def _read_response_line(self) -> str:
        """Read one response line from stdout with the configured timeout."""
        if self._process is None or self._process.stdout is None:
            raise RuntimeError("MCP client not started")

        result_queue: queue.Queue[str | BaseException] = queue.Queue(maxsize=1)

        def read_line() -> None:
            try:
                result_queue.put(self._process.stdout.readline())  # type: ignore[union-attr]
            except BaseException as exc:
                result_queue.put(exc)

        thread = threading.Thread(target=read_line, daemon=True)
        thread.start()

        try:
            result = result_queue.get(timeout=self._timeout)
        except queue.Empty:
            self.stop()
            raise TimeoutError(f"MCP request timed out after {self._timeout}s") from None

        if isinstance(result, BaseException):
            raise result
        return result

    def _send_notification(self, method: str, params: dict[str, Any]) -> None:
        """Send a JSON-RPC 2.0 notification (no response expected)."""
        if self._process is None or self._process.stdin is None:
            return

        notification = {
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        }
        payload = json.dumps(notification) + "\n"
        self._process.stdin.write(payload)
        self._process.stdin.flush()

    def __enter__(self) -> MCPClient:
        self.start()
        return self

    def __exit__(self, *exc: Any) -> None:
        self.stop()


class MCPToolAdapter(Tool):
    """Wraps a single MCP tool as a MiniClaw ``Tool``.

    Args:
        mcp_tool_name: The tool name on the MCP server.
        mcp_tool_schema: The tool's JSON Schema from the MCP server.
        mcp_client: The MCP client to call the tool on.
        description: Optional override for the tool description.
    """

    def __init__(
        self,
        mcp_tool_name: str,
        mcp_tool_schema: dict[str, Any],
        mcp_client: MCPClient,
        description: str | None = None,
    ) -> None:
        self.name = mcp_tool_name
        self.description = description or mcp_tool_schema.get(
            "description", f"MCP tool: {mcp_tool_name}"
        )
        self.schema = _convert_schema(mcp_tool_schema)
        self._client = mcp_client

    def run(self, **kwargs: Any) -> Any:
        """Call the MCP tool via the client."""
        return self._client.call_tool(self.name, kwargs)


class MCPToolRegistry:
    """Discovers and registers tools from an MCP server.

    Args:
        server_config: MCP server configuration.  Must contain at least
            ``{"command": "...", "args": [...]}``.
        timeout: Timeout for MCP requests.

    Usage::

        registry = MCPToolRegistry({"command": "my-mcp-server", "args": []})
        tools = registry.discover_tools()
        for tool in tools:
            agent_registry.register(tool)
    """

    def __init__(
        self,
        server_config: dict[str, Any],
        timeout: float = 30.0,
    ) -> None:
        self._config = server_config
        self._timeout = timeout
        self._client: MCPClient | None = None

    def discover_tools(self) -> list[MCPToolAdapter]:
        """Connect to the MCP server and discover available tools.

        Returns:
            List of ``MCPToolAdapter`` instances ready for registration.
        """
        command = self._config.get("command")
        if not command:
            raise ValueError("MCP server config must include 'command'")

        args = self._config.get("args", [])
        self._client = MCPClient(command, args=args, timeout=self._timeout)
        self._client.start()

        mcp_tools = self._client.list_tools()
        adapters: list[MCPToolAdapter] = []

        for tool_def in mcp_tools:
            name = tool_def.get("name", "")
            if not name:
                continue
            adapter = MCPToolAdapter(
                mcp_tool_name=name,
                mcp_tool_schema=tool_def,
                mcp_client=self._client,
                description=tool_def.get("description"),
            )
            adapters.append(adapter)

        return adapters

    def register_all(self, registry: Any) -> int:
        """Discover MCP tools and register them with a ToolRegistry.

        Args:
            registry: A ``ToolRegistry`` instance.

        Returns:
            Number of tools registered.
        """
        tools = self.discover_tools()
        for tool in tools:
            registry.register(tool)
        return len(tools)

    def close(self) -> None:
        """Stop the MCP client."""
        if self._client is not None:
            self._client.stop()
            self._client = None


def _convert_schema(mcp_schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an MCP tool input schema to a standard JSON Schema.

    MCP tools provide ``inputSchema`` which is already a JSON Schema.
    This function extracts it and ensures it's well-formed.
    """
    # MCP tools wrap their schema in "inputSchema"
    input_schema = mcp_schema.get("inputSchema", mcp_schema)

    # Ensure it's a valid JSON Schema object
    if not isinstance(input_schema, dict):
        return {"type": "object", "properties": {}}

    # Ensure it has a type
    if "type" not in input_schema:
        input_schema["type"] = "object"

    return input_schema
