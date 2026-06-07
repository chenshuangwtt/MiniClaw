"""Web search tool for MiniClaw.

Uses a simple HTTP-based search API. When this tool is run through
``ToolExecutor``, it is gated by ``PermissionPolicy`` and should only
be registered when ``allow_search`` is enabled.

Usage::

    from miniclaw.tools.search_tool import WebSearch
    from miniclaw.tools.permissions import PermissionPolicy

    # Enable search in the permission policy
    policy = PermissionPolicy(allow_search=True)

    tool = WebSearch()
    result = tool.run(query="Python asyncio tutorial", max_results=3)
"""

from __future__ import annotations

import json
import logging
import urllib.request
import urllib.parse
from typing import Any

from miniclaw.tools.base import Tool

logger = logging.getLogger(__name__)

# Maximum results per search
MAX_RESULTS = 10


class WebSearch(Tool):
    """Search the web for information.

    This tool performs an HTTP request to a search endpoint. Register it
    only when ``PermissionPolicy.allow_search`` is enabled.

    Attributes:
        name: ``"web_search"``
        description: Human-readable description for the LLM.
        schema: JSON Schema for the tool's parameters.
    """

    name = "web_search"
    description = (
        "Search the web for information. "
        "Returns a list of search results with title, URL, and snippet."
    )
    schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum number of results (default: 5, max: {MAX_RESULTS}).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    def run(self, query: str, max_results: int = 5, **kwargs: Any) -> dict[str, Any]:
        """Execute a web search.

        Args:
            query: The search query string.
            max_results: Maximum results to return.

        Returns:
            A dict with ``results`` (list of {title, url, snippet})
            or ``error`` on failure.
        """
        max_results = min(max_results, MAX_RESULTS)

        try:
            results = _search_ddg(query, max_results)
            return {"query": query, "results": results, "count": len(results)}
        except Exception as exc:
            logger.warning("Web search failed: %s", exc)
            return {"error": f"Search failed: {exc}"}


def _search_ddg(query: str, max_results: int) -> list[dict[str, str]]:
    """Search using DuckDuckGo's instant answer API.

    Falls back to an empty list if the API is unavailable.
    """
    url = "https://api.duckduckgo.com/"
    params = urllib.parse.urlencode(
        {
            "q": query,
            "format": "json",
            "no_html": 1,
            "skip_disambig": 1,
        }
    )
    full_url = f"{url}?{params}"

    req = urllib.request.Request(
        full_url,
        headers={"User-Agent": "MiniClaw/0.4"},
    )

    try:
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception:
        return []

    results: list[dict[str, str]] = []

    # Instant answer
    abstract = data.get("AbstractText", "")
    abstract_url = data.get("AbstractURL", "")
    if abstract:
        results.append(
            {
                "title": data.get("Heading", query),
                "url": abstract_url,
                "snippet": abstract[:300],
            }
        )

    # Related topics
    for topic in data.get("RelatedTopics", []):
        if len(results) >= max_results:
            break
        if isinstance(topic, dict) and "Text" in topic:
            results.append(
                {
                    "title": topic.get("Text", "")[:100],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", "")[:300],
                }
            )

    return results[:max_results]
