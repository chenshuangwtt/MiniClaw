"""OpenAPI tool adapter — generate MiniClaw tools from an OpenAPI 3.x spec.

Parses an OpenAPI specification and creates ``Tool`` instances for each
operation, so the agent can call REST APIs without hand-written tools.

Supports:
    - GET, POST, PUT, DELETE, PATCH
    - Path, query, and header parameters
    - JSON request bodies
    - Bearer token authentication

Usage::

    from miniclaw.tools.openapi_adapter import OpenAPIToolRegistry

    registry = OpenAPIToolRegistry("https://petstore3.swagger.io/api/v3/openapi.json")
    tools = registry.discover_tools()
    for tool in tools:
        agent_registry.register(tool)
"""

from __future__ import annotations

import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

import yaml

from miniclaw.tools.base import Tool

logger = logging.getLogger(__name__)


class OpenAPIToolAdapter(Tool):
    """Wraps a single OpenAPI operation as a MiniClaw ``Tool``.

    Args:
        operation_id: The OpenAPI operationId (used as tool name).
        method: HTTP method (GET, POST, etc.).
        path: URL path template (e.g., ``/pets/{petId}``).
        summary: Human-readable description.
        parameters: OpenAPI parameter definitions.
        request_body: OpenAPI requestBody schema (for POST/PUT/PATCH).
        base_url: The API base URL.
        auth_token: Optional Bearer token for authentication.
    """

    def __init__(
        self,
        operation_id: str,
        method: str,
        path: str,
        summary: str,
        parameters: list[dict[str, Any]],
        request_body: dict[str, Any] | None,
        base_url: str,
        auth_token: str | None = None,
    ) -> None:
        self.name = operation_id
        self.description = summary or f"{method.upper()} {path}"
        self.schema = _build_json_schema(parameters, request_body)
        self._method = method.upper()
        self._path = path
        self._base_url = base_url.rstrip("/")
        self._parameters = parameters
        self._request_body = request_body
        self._auth_token = auth_token

    def run(self, **kwargs: Any) -> Any:
        """Execute the API call.

        Path parameters are substituted into the URL template.
        Query parameters are appended as URL query string.
        Header parameters are set as HTTP headers.
        Request body is sent as JSON (for POST/PUT/PATCH).

        Returns:
            A dict with ``status``, ``headers``, and ``body`` (parsed JSON
            or raw text).
        """
        url = self._build_url(kwargs)
        body = self._build_body(kwargs)
        headers = self._build_headers(kwargs, has_body=body is not None)

        try:
            data = json.dumps(body).encode("utf-8") if body else None
            req = urllib.request.Request(
                url,
                data=data,
                headers=headers,
                method=self._method,
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                response_body = resp.read().decode("utf-8", errors="replace")
                try:
                    parsed = json.loads(response_body)
                except json.JSONDecodeError:
                    parsed = response_body
                return {
                    "status": resp.status,
                    "body": parsed,
                }
        except urllib.error.HTTPError as exc:
            error_body = exc.read().decode("utf-8", errors="replace") if exc.fp else ""
            return {
                "status": exc.code,
                "error": exc.reason,
                "body": error_body[:2000],
            }
        except Exception as exc:
            return {"error": f"{type(exc).__name__}: {exc}"}

    def _build_url(self, kwargs: dict[str, Any]) -> str:
        """Build the full URL with path and query parameters."""
        path = self._path
        query_parts: list[str] = []

        for param in self._parameters:
            name = param.get("name", "")
            location = param.get("in", "")
            value = kwargs.get(name)

            if value is None:
                continue

            if location == "path":
                path = path.replace(f"{{{name}}}", urllib.parse.quote(str(value), safe=""))
            elif location == "query":
                query_parts.append(
                    f"{urllib.parse.quote(str(name), safe='')}="
                    f"{urllib.parse.quote(str(value), safe='')}"
                )

        url = f"{self._base_url}{path}"
        if query_parts:
            url += "?" + "&".join(query_parts)
        return url

    def _build_headers(self, kwargs: dict[str, Any], has_body: bool = False) -> dict[str, str]:
        """Build HTTP headers from parameters and auth."""
        headers: dict[str, str] = {"Accept": "application/json"}

        for param in self._parameters:
            name = param.get("name", "")
            if param.get("in") == "header" and name in kwargs:
                headers[name] = str(kwargs[name])

        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"

        if has_body:
            headers["Content-Type"] = "application/json"

        return headers

    def _build_body(self, kwargs: dict[str, Any]) -> Any:
        """Extract the request body from kwargs."""
        if self._method not in ("POST", "PUT", "PATCH"):
            return None

        if self._request_body is None:
            return None

        # Find the body parameter name from the schema
        schema = _extract_request_schema(self._request_body)
        if schema and "properties" in schema:
            body: dict[str, Any] = {}
            for prop_name in schema["properties"]:
                if prop_name in kwargs:
                    body[prop_name] = kwargs[prop_name]
            return body if body else None

        return None


class OpenAPIToolRegistry:
    """Discovers and registers tools from an OpenAPI 3.x specification.

    Args:
        spec_url_or_path: URL or file path to the OpenAPI spec (JSON or YAML).
        base_url: Override the base URL from the spec.
        auth_token: Optional Bearer token for all operations.

    Usage::

        registry = OpenAPIToolRegistry("https://api.example.com/openapi.json")
        tools = registry.discover_tools()
    """

    def __init__(
        self,
        spec_url_or_path: str,
        base_url: str | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._spec_url = spec_url_or_path
        self._base_url_override = base_url
        self._auth_token = auth_token
        self._spec: dict[str, Any] | None = None

    def load_spec(self) -> dict[str, Any]:
        """Load and parse the OpenAPI specification.

        Returns:
            The parsed spec as a dict.
        """
        if self._spec is not None:
            return self._spec

        if self._spec_url.startswith(("http://", "https://")):
            req = urllib.request.Request(self._spec_url)
            with urllib.request.urlopen(req, timeout=30) as resp:
                raw = resp.read().decode("utf-8")
        else:
            path = Path(self._spec_url)
            raw = path.read_text(encoding="utf-8")

        self._spec = _parse_spec(raw, self._spec_url)

        return self._spec

    def discover_tools(self) -> list[OpenAPIToolAdapter]:
        """Parse the spec and create tool adapters for each operation.

        Returns:
            List of ``OpenAPIToolAdapter`` instances.
        """
        spec = self.load_spec()
        base_url = self._base_url_override or _extract_base_url(spec)
        if not base_url:
            logger.warning("No base URL found in spec and none provided.")
            return []

        tools: list[OpenAPIToolAdapter] = []
        paths = spec.get("paths", {})

        for path, path_item in paths.items():
            for method in ("get", "post", "put", "delete", "patch"):
                operation = path_item.get(method)
                if not operation:
                    continue

                op_id = operation.get("operationId", f"{method}_{_slugify(path)}")
                summary = operation.get("summary", "") or operation.get("description", "")
                parameters = _merge_parameters(
                    path_item.get("parameters", []),
                    operation.get("parameters", []),
                )
                request_body = operation.get("requestBody")

                tool = OpenAPIToolAdapter(
                    operation_id=_sanitize_name(op_id),
                    method=method,
                    path=path,
                    summary=summary,
                    parameters=parameters,
                    request_body=request_body,
                    base_url=base_url,
                    auth_token=self._auth_token,
                )
                tools.append(tool)

        return tools

    def register_all(self, registry: Any) -> int:
        """Discover API operations and register them with a ToolRegistry.

        Args:
            registry: A ``ToolRegistry`` instance.

        Returns:
            Number of tools registered.
        """
        tools = self.discover_tools()
        for tool in tools:
            registry.register(tool)
        return len(tools)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _extract_base_url(spec: dict[str, Any]) -> str:
    """Extract the base URL from an OpenAPI spec."""
    # OpenAPI 3.x: servers[0].url
    servers = spec.get("servers", [])
    if servers and isinstance(servers, list):
        url = servers[0].get("url", "")
        if url:
            return url.rstrip("/")

    # Swagger 2.0: host + basePath + schemes
    host = spec.get("host", "")
    base_path = spec.get("basePath", "")
    schemes = spec.get("schemes", ["https"])
    if host:
        scheme = schemes[0] if schemes else "https"
        return f"{scheme}://{host}{base_path}".rstrip("/")

    return ""


def _parse_spec(raw: str, source: str = "") -> dict[str, Any]:
    """Parse an OpenAPI spec from JSON or YAML text."""
    source_lower = source.lower()
    if source_lower.endswith((".yaml", ".yml")):
        parsed = yaml.safe_load(raw)
    else:
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            parsed = yaml.safe_load(raw)

    if not isinstance(parsed, dict):
        raise ValueError("OpenAPI specification must parse to an object")
    return parsed


def _merge_parameters(
    path_params: list[dict[str, Any]],
    operation_params: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge path-level and operation-level parameters.

    Operation-level parameters override path-level ones with the same
    name and location.
    """
    merged: dict[tuple[str, str], dict[str, Any]] = {}

    for param in path_params:
        key = (param.get("name", ""), param.get("in", ""))
        merged[key] = param

    for param in operation_params:
        key = (param.get("name", ""), param.get("in", ""))
        merged[key] = param

    return list(merged.values())


def _build_json_schema(
    parameters: list[dict[str, Any]],
    request_body: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build a JSON Schema from OpenAPI parameters and request body."""
    properties: dict[str, Any] = {}
    required: list[str] = []

    # Parameters (path, query, header)
    for param in parameters:
        name = param.get("name", "")
        if not name:
            continue
        schema = param.get("schema", {"type": "string"})
        properties[name] = _convert_openapi_schema(schema)
        if param.get("required", False):
            required.append(name)

    # Request body
    if request_body:
        body_schema = _extract_request_schema(request_body)
        if body_schema and "properties" in body_schema:
            for prop_name, prop_schema in body_schema["properties"].items():
                properties[prop_name] = _convert_openapi_schema(prop_schema)
            body_required = body_schema.get("required", [])
            required.extend(body_required)

    result: dict[str, Any] = {
        "type": "object",
        "properties": properties,
    }
    if required:
        result["required"] = list(set(required))

    return result


def _extract_request_schema(request_body: dict[str, Any]) -> dict[str, Any] | None:
    """Extract the JSON schema from an OpenAPI requestBody."""
    content = request_body.get("content", {})
    json_content = content.get("application/json", {})
    return json_content.get("schema")


def _convert_openapi_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Convert an OpenAPI schema to a standard JSON Schema."""
    result: dict[str, Any] = {}
    for key in ("type", "description", "enum", "default", "format", "items"):
        if key in schema:
            result[key] = schema[key]
    if "type" not in result:
        result["type"] = "string"
    return result


def _sanitize_name(name: str) -> str:
    """Sanitize an operationId to be a valid tool name."""
    # Replace non-alphanumeric chars with underscores
    sanitized = re.sub(r"[^a-zA-Z0-9_]", "_", name)
    # Remove consecutive underscores
    sanitized = re.sub(r"_+", "_", sanitized)
    # Strip leading/trailing underscores
    return sanitized.strip("_").lower()


def _slugify(path: str) -> str:
    """Convert a URL path to a slug for generating operationIds."""
    slug = path.replace("/", "_").replace("{", "").replace("}", "")
    return re.sub(r"[^a-zA-Z0-9_]", "_", slug).strip("_").lower()
