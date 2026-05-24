"""RAG search tool — wraps the LightRAG index built by ingest_rag.ipynb.

Returns retrieved context only (no generation) so the calling subagent can
synthesize its own answer. Loading the LightRAG storage is done lazily on
first call and cached across subsequent calls.
"""

from __future__ import annotations

import json
from typing import Any

import httpx
import numpy as np
from claude_agent_sdk import tool
from lightrag import LightRAG, QueryParam
from lightrag.utils import wrap_embedding_func_with_attrs

from legal_explainer.agents.config import (
    OLLAMA_BASE_URL,
    OLLAMA_EMBED_DIM,
    OLLAMA_EMBED_MAX_TOKENS,
    OLLAMA_EMBED_MODEL,
    RAG_DEFAULT_MODE,
    RAG_DEFAULT_TOP_K,
    RAG_LLM_MODEL,
    RAG_WORKING_DIR,
)
from legal_explainer.agents.tools.keyword_extract import extract_keywords


@wrap_embedding_func_with_attrs(
    embedding_dim=OLLAMA_EMBED_DIM,
    max_token_size=OLLAMA_EMBED_MAX_TOKENS,
    model_name=OLLAMA_EMBED_MODEL,
)
async def _embed(texts: list[str]) -> np.ndarray:
    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{OLLAMA_BASE_URL}/api/embed",
            json={"model": OLLAMA_EMBED_MODEL, "input": texts},
        )
        resp.raise_for_status()
    return np.array(resp.json()["embeddings"], dtype=np.float32)


async def _llm_guard(*args, **kwargs) -> str:
    """LightRAG requires an llm_model_func to be set. We never call the LLM
    from this tool — keyword extraction is pre-computed and passed via
    QueryParam(hl_keywords=..., ll_keywords=...) so LightRAG's LLM extractor
    is bypassed entirely (see lightrag/operate.py:3316). If this ever fires,
    something is calling LightRAG with a path that needs generation, which
    this tool is not designed to do.
    """
    raise RuntimeError(
        "search_legal_documents must run with pre-extracted keywords. "
        "If this fires, the caller invoked a LightRAG path that needs LLM "
        "generation; use the Researcher subagent for synthesis instead."
    )


_rag_instance: LightRAG | None = None


async def get_rag() -> LightRAG:
    """Lazy-load the LightRAG instance against the on-disk index. The first
    call pays the storage init cost; subsequent calls return the cached handle."""
    global _rag_instance
    if _rag_instance is not None:
        return _rag_instance

    if not RAG_WORKING_DIR.exists():
        raise FileNotFoundError(
            f"LightRAG storage not found at {RAG_WORKING_DIR}. "
            f"Run ingest_rag.ipynb first."
        )

    rag = LightRAG(
        working_dir=str(RAG_WORKING_DIR),
        llm_model_func=_llm_guard,
        llm_model_name=RAG_LLM_MODEL,
        embedding_func=_embed,
        llm_model_max_async=4,
        embedding_func_max_async=4,
        default_embedding_timeout=180,
        default_llm_timeout=600,
        chunk_token_size=4000,
        chunk_overlap_token_size=200,
        addon_params={"language": "Arabic and English"},
    )
    await rag.initialize_storages()
    _rag_instance = rag
    return rag


async def search_legal_context(query: str) -> dict[str, Any]:
    """Pure-Python entrypoint — call this directly from notebooks/tests.
    The @tool-decorated `search_legal_documents` is a thin wrapper around it
    for SDK MCP dispatch."""
    rag = await get_rag()

    hl_keywords, ll_keywords = extract_keywords(query)

    result = await rag.aquery_data(
        query,
        param=QueryParam(
            mode=RAG_DEFAULT_MODE,
            top_k=RAG_DEFAULT_TOP_K,
            hl_keywords=hl_keywords,
            ll_keywords=ll_keywords,
            enable_rerank=False,
        ),
    )

    if result.get("status") != "success":
        return {
            "ok": False,
            "query": query,
            "message": result.get("message", "no results"),
        }

    data = result.get("data", {})
    return {
        "ok": True,
        "query": query,
        "entities": [
            {
                "name": e.get("entity_name"),
                "type": e.get("entity_type"),
                "description": e.get("description"),
            }
            for e in data.get("entities", [])[:10]
        ],
        "relationships": [
            {
                "src": r.get("src_id"),
                "tgt": r.get("tgt_id"),
                "description": r.get("description"),
                "keywords": r.get("keywords"),
            }
            for r in data.get("relationships", [])[:10]
        ],
        "chunks": [
            {"content": c.get("content"), "chunk_id": c.get("chunk_id")}
            for c in data.get("chunks", [])[:5]
        ],
    }


@tool(
    "search_legal_documents",
    (
        "Retrieve context from the Egyptian Civil Code knowledge graph for an "
        "open-ended legal question. Use this for questions about topics, "
        "concepts, or 'how does Egyptian law handle X?'. Do NOT use this when "
        "the user is asking for a specific article by number — use "
        "check_statute_reference instead. Returns entities, relationships, "
        "and source chunks for you to reason over. Returns {ok: false, ...} "
        "when nothing matched."
    ),
    {"query": str},
)
async def search_legal_documents(args: dict[str, Any]) -> dict[str, Any]:
    """SDK MCP wrapper around search_legal_context — packs the JSON payload
    into the MCP-tool envelope the SDK expects."""
    payload = await search_legal_context(args["query"])
    return {
        "content": [
            {"type": "text", "text": json.dumps(payload, ensure_ascii=False)}
        ]
    }