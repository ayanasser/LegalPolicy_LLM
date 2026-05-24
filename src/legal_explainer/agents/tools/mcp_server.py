"""Bundle every legal-explainer tool into a single MCP server.

Subagents and the orchestrator import `legal_tools_server` from here and
expose the subset they're permitted to use via `allowed_tools`.

Tool name convention when allow-listed:
    mcp__legal_tools__<tool_name>
e.g. mcp__legal_tools__get_legal_definition
"""

from __future__ import annotations

from claude_agent_sdk import create_sdk_mcp_server

from legal_explainer.agents.tools.glossary import get_legal_definition
from legal_explainer.agents.tools.rag_search import search_legal_documents
from legal_explainer.agents.tools.statute import check_statute_reference
from legal_explainer.agents.tools.web_search import web_search

SERVER_NAME = "legal_tools"

legal_tools_server = create_sdk_mcp_server(
    name=SERVER_NAME,
    version="0.1.0",
    tools=[
        get_legal_definition,
        check_statute_reference,
        search_legal_documents,
        web_search,
    ],
)

# Convenience: the canonical tool ids that ClaudeAgentOptions(allowed_tools=...)
# expects. Use these constants instead of stringly-typed literals.
TOOL_GET_DEFINITION = f"mcp__{SERVER_NAME}__get_legal_definition"
TOOL_CHECK_STATUTE = f"mcp__{SERVER_NAME}__check_statute_reference"
TOOL_SEARCH_DOCS = f"mcp__{SERVER_NAME}__search_legal_documents"
TOOL_WEB_SEARCH = f"mcp__{SERVER_NAME}__web_search"

ALL_TOOLS = [
    TOOL_GET_DEFINITION,
    TOOL_CHECK_STATUTE,
    TOOL_SEARCH_DOCS,
    TOOL_WEB_SEARCH,
]