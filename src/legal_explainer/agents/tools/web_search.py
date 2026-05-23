"""Web search tool — DuckDuckGo wrapper, model-agnostic.

The user picked Anthropic-native web_search in the design phase, but that's a
server-side tool only available when calling Claude models directly. Since this
project's default route is `glm-5` via a proxy, we ship a small DuckDuckGo
client so web_search works regardless of which model the subagent uses.

To swap to Anthropic-native later: enable `tools={"type": "preset", "preset":
"claude_code"}` and `allowed_tools=["WebSearch"]` in the orchestrator's
ClaudeAgentOptions, drop this file from the MCP server, and the model will
call Anthropic's built-in tool instead.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
from claude_agent_sdk import tool


async def _ddg_search(query: str, max_results: int) -> list[dict[str, str]]:
    """Hit DuckDuckGo's lite HTML endpoint and parse results.

    Uses the /html/ endpoint which doesn't require an API key. Returns a list
    of {title, url, snippet} dicts.
    """
    # Lazy import — only needed when the tool actually runs.
    try:
        from duckduckgo_search import DDGS  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "duckduckgo-search not installed. Install with: "
            "`pip install duckduckgo-search`"
        ) from e

    results: list[dict[str, str]] = []
    with DDGS() as ddgs:
        for r in ddgs.text(query, max_results=max_results):
            results.append(
                {
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                }
            )
    return results


@tool(
    "web_search",
    (
        "Search the public web for current information when the user asks "
        "about recent events, amendments, news, or anything that is unlikely "
        "to be in the local Egyptian Civil Code corpus. Do NOT use for "
        "questions about specific articles or general legal concepts — use "
        "check_statute_reference or search_legal_documents for those. "
        "Returns a list of {title, url, snippet} dicts; cite the URLs in the "
        "final answer."
    ),
    {"query": str, "max_results": int},
)
async def web_search(args: dict[str, Any]) -> dict[str, Any]:
    """MCP tool wrapper around DuckDuckGo search."""
    max_results = args.get("max_results") or 5
    try:
        results = await _ddg_search(args["query"], max_results)
    except Exception as e:
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps(
                        {
                            "ok": False,
                            "query": args["query"],
                            "error": str(e),
                            "message": "Web search unavailable — answer from local corpus only.",
                        },
                        ensure_ascii=False,
                    ),
                }
            ]
        }
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    {"ok": True, "query": args["query"], "results": results},
                    ensure_ascii=False,
                ),
            }
        ]
    }