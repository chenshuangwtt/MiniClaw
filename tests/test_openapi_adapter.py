"""Tests for OpenAPI tool adapter."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path


from miniclaw.tools.openapi_adapter import (
    OpenAPIToolAdapter,
    OpenAPIToolRegistry,
    _build_json_schema,
    _convert_openapi_schema,
    _extract_base_url,
    _merge_parameters,
    _sanitize_name,
    _slugify,
)
from miniclaw.tools.registry import ToolRegistry


# ---------------------------------------------------------------------------
# Sample spec
# ---------------------------------------------------------------------------

SAMPLE_SPEC = {
    "openapi": "3.0.0",
    "info": {"title": "Pet Store", "version": "1.0.0"},
    "servers": [{"url": "https://petstore.example.com/api/v1"}],
    "paths": {
        "/pets": {
            "get": {
                "operationId": "listPets",
                "summary": "List all pets",
                "parameters": [
                    {
                        "name": "limit",
                        "in": "query",
                        "schema": {"type": "integer"},
                        "required": False,
                    },
                ],
                "responses": {"200": {"description": "A list of pets"}},
            },
            "post": {
                "operationId": "createPet",
                "summary": "Create a pet",
                "requestBody": {
                    "content": {
                        "application/json": {
                            "schema": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "tag": {"type": "string"},
                                },
                                "required": ["name"],
                            }
                        }
                    }
                },
                "responses": {"201": {"description": "Pet created"}},
            },
        },
        "/pets/{petId}": {
            "get": {
                "operationId": "getPet",
                "summary": "Get a pet by ID",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"200": {"description": "A pet"}},
            },
            "delete": {
                "operationId": "deletePet",
                "summary": "Delete a pet",
                "parameters": [
                    {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
                ],
                "responses": {"204": {"description": "Deleted"}},
            },
        },
    },
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSlugify:
    def test_basic(self) -> None:
        assert _slugify("/pets") == "pets"

    def test_with_param(self) -> None:
        assert _slugify("/pets/{petId}") == "pets_petid"

    def test_nested(self) -> None:
        assert _slugify("/api/v1/users") == "api_v1_users"


class TestSanitizeName:
    def test_basic(self) -> None:
        assert _sanitize_name("listPets") == "listpets"

    def test_with_dashes(self) -> None:
        assert _sanitize_name("my-operation") == "my_operation"

    def test_with_spaces(self) -> None:
        assert _sanitize_name("my operation") == "my_operation"


class TestExtractBaseUrl:
    def test_openapi3(self) -> None:
        spec = {"servers": [{"url": "https://api.example.com/v2"}]}
        assert _extract_base_url(spec) == "https://api.example.com/v2"

    def test_swagger2(self) -> None:
        spec = {"host": "api.example.com", "basePath": "/v2", "schemes": ["https"]}
        assert _extract_base_url(spec) == "https://api.example.com/v2"

    def test_empty(self) -> None:
        assert _extract_base_url({}) == ""


class TestMergeParameters:
    def test_merge_no_overlap(self) -> None:
        path_params = [{"name": "petId", "in": "path"}]
        op_params = [{"name": "limit", "in": "query"}]
        result = _merge_parameters(path_params, op_params)
        assert len(result) == 2

    def test_merge_with_override(self) -> None:
        path_params = [{"name": "limit", "in": "query", "schema": {"type": "integer"}}]
        op_params = [{"name": "limit", "in": "query", "schema": {"type": "string"}}]
        result = _merge_parameters(path_params, op_params)
        assert len(result) == 1
        assert result[0]["schema"]["type"] == "string"


class TestConvertOpenAPISchema:
    def test_basic(self) -> None:
        schema = {"type": "string", "description": "A name"}
        result = _convert_openapi_schema(schema)
        assert result["type"] == "string"
        assert result["description"] == "A name"

    def test_missing_type(self) -> None:
        result = _convert_openapi_schema({"description": "desc"})
        assert result["type"] == "string"


class TestBuildJsonSchema:
    def test_parameters_only(self) -> None:
        params = [
            {"name": "limit", "in": "query", "schema": {"type": "integer"}},
            {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}},
        ]
        result = _build_json_schema(params, None)
        assert "limit" in result["properties"]
        assert "petId" in result["properties"]
        assert "petId" in result["required"]

    def test_with_request_body(self) -> None:
        request_body = {
            "content": {
                "application/json": {
                    "schema": {
                        "type": "object",
                        "properties": {"name": {"type": "string"}},
                        "required": ["name"],
                    }
                }
            }
        }
        result = _build_json_schema([], request_body)
        assert "name" in result["properties"]
        assert "name" in result["required"]


# ---------------------------------------------------------------------------
# OpenAPIToolAdapter
# ---------------------------------------------------------------------------


class TestOpenAPIToolAdapter:
    def test_basic_properties(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="listPets",
            method="get",
            path="/pets",
            summary="List all pets",
            parameters=[{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
            request_body=None,
            base_url="https://api.example.com",
        )
        assert tool.name == "listPets"
        assert "List all pets" in tool.description
        assert "limit" in tool.schema["properties"]

    def test_build_url_with_query(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="listPets",
            method="get",
            path="/pets",
            summary="",
            parameters=[{"name": "limit", "in": "query", "schema": {"type": "integer"}}],
            request_body=None,
            base_url="https://api.example.com",
        )
        url = tool._build_url({"limit": 10})
        assert "limit=10" in url
        assert url.startswith("https://api.example.com/pets")

    def test_build_url_with_path_param(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="getPet",
            method="get",
            path="/pets/{petId}",
            summary="",
            parameters=[
                {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            request_body=None,
            base_url="https://api.example.com",
        )
        url = tool._build_url({"petId": "42"})
        assert url == "https://api.example.com/pets/42"

    def test_build_url_encodes_path_param(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="getPet",
            method="get",
            path="/pets/{petId}",
            summary="",
            parameters=[
                {"name": "petId", "in": "path", "required": True, "schema": {"type": "string"}}
            ],
            request_body=None,
            base_url="https://api.example.com",
        )
        url = tool._build_url({"petId": "a/b c?x=1"})
        assert url == "https://api.example.com/pets/a%2Fb%20c%3Fx%3D1"

    def test_build_headers_with_auth(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="test",
            method="get",
            path="/test",
            summary="",
            parameters=[],
            request_body=None,
            base_url="https://api.example.com",
            auth_token="my-token",
        )
        headers = tool._build_headers({})
        assert headers["Authorization"] == "Bearer my-token"

    def test_build_headers_with_json_body_content_type(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="test",
            method="post",
            path="/test",
            summary="",
            parameters=[],
            request_body=None,
            base_url="https://api.example.com",
        )
        headers = tool._build_headers({}, has_body=True)
        assert headers["Content-Type"] == "application/json"

    def test_build_body_for_post(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="createPet",
            method="post",
            path="/pets",
            summary="",
            parameters=[],
            request_body={
                "content": {
                    "application/json": {
                        "schema": {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                        }
                    }
                }
            },
            base_url="https://api.example.com",
        )
        body = tool._build_body({"name": "Fido"})
        assert body == {"name": "Fido"}

    def test_build_body_none_for_get(self) -> None:
        tool = OpenAPIToolAdapter(
            operation_id="test",
            method="get",
            path="/test",
            summary="",
            parameters=[],
            request_body=None,
            base_url="https://api.example.com",
        )
        assert tool._build_body({}) is None


# ---------------------------------------------------------------------------
# OpenAPIToolRegistry
# ---------------------------------------------------------------------------


class TestOpenAPIToolRegistry:
    def test_discover_tools_from_file(self) -> None:
        """Test discovering tools from a local JSON spec file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_SPEC, f)
            spec_path = f.name

        try:
            registry = OpenAPIToolRegistry(spec_path)
            tools = registry.discover_tools()
            assert len(tools) == 4  # listPets, createPet, getPet, deletePet
            names = {t.name for t in tools}
            assert "listpets" in names
            assert "createpet" in names
        finally:
            Path(spec_path).unlink()

    def test_discover_tools_from_yaml_file(self) -> None:
        """Test discovering tools from a local YAML spec file."""
        yaml_spec = """
openapi: 3.0.0
info:
  title: Test API
  version: 1.0.0
servers:
  - url: https://api.example.com
paths:
  /status:
    get:
      operationId: getStatus
      summary: Get service status
      responses:
        "200":
          description: OK
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_spec)
            spec_path = f.name

        try:
            registry = OpenAPIToolRegistry(spec_path)
            tools = registry.discover_tools()
            assert len(tools) == 1
            assert tools[0].name == "getstatus"
        finally:
            Path(spec_path).unlink()

    def test_register_all(self) -> None:
        """Test registering tools with a ToolRegistry."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_SPEC, f)
            spec_path = f.name

        try:
            api_registry = OpenAPIToolRegistry(spec_path)
            tool_registry = ToolRegistry()
            count = api_registry.register_all(tool_registry)
            assert count == 4
            assert tool_registry.get("listpets") is not None
        finally:
            Path(spec_path).unlink()

    def test_custom_base_url(self) -> None:
        """Test overriding the base URL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_SPEC, f)
            spec_path = f.name

        try:
            registry = OpenAPIToolRegistry(spec_path, base_url="https://custom.api.com")
            tools = registry.discover_tools()
            # Check that the custom base URL is used
            for tool in tools:
                assert tool._base_url == "https://custom.api.com"
        finally:
            Path(spec_path).unlink()

    def test_with_auth_token(self) -> None:
        """Test auth token propagation."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(SAMPLE_SPEC, f)
            spec_path = f.name

        try:
            registry = OpenAPIToolRegistry(spec_path, auth_token="test-token")
            tools = registry.discover_tools()
            for tool in tools:
                assert tool._auth_token == "test-token"
        finally:
            Path(spec_path).unlink()
