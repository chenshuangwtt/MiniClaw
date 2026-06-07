"""Tests for tools/search_tool.py."""

from unittest.mock import patch

from miniclaw.tools.search_tool import WebSearch
from miniclaw.tools.permissions import PermissionPolicy


class TestWebSearchSchema:
    def test_name(self):
        assert WebSearch.name == "web_search"

    def test_schema_has_query(self):
        assert "query" in WebSearch.schema["properties"]
        assert "query" in WebSearch.schema["required"]

    def test_schema_has_max_results(self):
        assert "max_results" in WebSearch.schema["properties"]


class TestWebSearchPermission:
    def test_blocked_by_default(self):
        policy = PermissionPolicy()
        decision = policy.check("web_search", {"query": "test"})
        assert decision.allowed is False
        assert "disabled" in decision.reason.lower()

    def test_allowed_when_enabled(self):
        policy = PermissionPolicy(allow_search=True)
        decision = policy.check("web_search", {"query": "test"})
        assert decision.allowed is True

    def test_allowed_via_approval(self):
        policy = PermissionPolicy(
            approval_required_tools={"web_search"},
            approval_callback=lambda name, args: True,
        )
        decision = policy.check("web_search", {"query": "test"})
        assert decision.allowed is True


class TestWebSearchRun:
    @patch("miniclaw.tools.search_tool._search_ddg")
    def test_returns_results(self, mock_search):
        mock_search.return_value = [
            {"title": "Python", "url": "https://python.org", "snippet": "Python is a language."},
        ]
        tool = WebSearch()
        result = tool.run(query="Python")
        assert result["count"] == 1
        assert result["results"][0]["title"] == "Python"

    @patch("miniclaw.tools.search_tool._search_ddg")
    def test_max_results_capped(self, mock_search):
        mock_search.return_value = [{"title": f"r{i}", "url": "", "snippet": ""} for i in range(20)]
        tool = WebSearch()
        tool.run(query="test", max_results=100)
        # Should call _search_ddg with max_results capped at MAX_RESULTS
        call_args = mock_search.call_args
        assert call_args[0][1] <= 10  # MAX_RESULTS

    @patch("miniclaw.tools.search_tool._search_ddg")
    def test_search_failure_returns_error(self, mock_search):
        mock_search.side_effect = Exception("network error")
        tool = WebSearch()
        result = tool.run(query="test")
        assert "error" in result
        assert "network error" in result["error"]

    @patch("miniclaw.tools.search_tool._search_ddg")
    def test_empty_results(self, mock_search):
        mock_search.return_value = []
        tool = WebSearch()
        result = tool.run(query="obscure query")
        assert result["count"] == 0
        assert result["results"] == []
