"""
Egyptian Civil Law — RAG API
=============================
Run:
    uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 --reload

Endpoints:
    GET  /health                  → liveness + readiness check
    POST /api/v1/ask              → full RAG (question → answer + sources)
    POST /api/v1/search           → retrieval only (no LLM answer)
    GET  /api/v1/article/{number} → fetch a single article by number
    GET  /docs                    → auto-generated Swagger UI
    GET  /redoc                   → ReDoc UI
"""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import RAGPipeline


# ─────────────────────────────────────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ─────────────────────────────────────────────────────────────────────────────

_pipeline: RAGPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    cfg = get_settings()
    _pipeline = RAGPipeline(cfg)
    _pipeline.startup()          # loads BGE-M3, opens Neo4j + Ollama connections
    yield
    _pipeline.shutdown()         # closes Neo4j driver cleanly


def get_pipeline() -> RAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — server still starting up.")
    return _pipeline


# ─────────────────────────────────────────────────────────────────────────────
# App
# ─────────────────────────────────────────────────────────────────────────────

cfg = get_settings()

app = FastAPI(
    title="Egyptian Civil Law RAG API",
    description=(
        "Retrieval-Augmented Generation over 1 093 Egyptian Civil Law articles.\n\n"
        "**Stack:** BAAI/bge-m3 embeddings · Neo4j Aura vector index · "
        "Qwen3:4b (Ollama) for metadata extraction and answer generation."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=cfg.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Global error handler
# ─────────────────────────────────────────────────────────────────────────────

@app.exception_handler(Exception)
async def generic_error_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": str(exc)},
    )


# ─────────────────────────────────────────────────────────────────────────────
# Request / Response schemas
# ─────────────────────────────────────────────────────────────────────────────

class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["What are the conditions under which exercising a right becomes unlawful?"],
    )
    top_k: int = Field(5, ge=1, le=20, description="Number of source articles to retrieve")


class SearchRequest(BaseModel):
    query: str = Field(
        ...,
        min_length=3,
        max_length=2000,
        examples=["prescription and limitation periods"],
    )
    top_k: int = Field(10, ge=1, le=50, description="Maximum articles to return")


class ArticleOut(BaseModel):
    id:      str
    number:  int
    english: str
    arabic:  str
    score:   float = 0.0


class MetadataOut(BaseModel):
    keywords_en:     list[str] = []
    keywords_ar:     list[str] = []
    legal_topics:    list[str] = []
    article_numbers: list[int] = []
    search_query:    str       = ""


class GenerationInfo(BaseModel):
    """The answer model + its sampling params (for observability traces)."""
    model:        str
    params:       dict = {}
    embed_model:  str = ""


class AskResponse(BaseModel):
    answer:             str
    articles:           list[ArticleOut]
    metadata:           MetadataOut
    processing_time_ms: int
    generation:         GenerationInfo | None = None


class HealthResponse(BaseModel):
    status:     str           # "ok" | "degraded" | "error"
    neo4j:      str           # "connected" | "error: ..."
    ollama:     str           # "ready" | "error: ..."
    llm_model:  str
    embed_model: str


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthResponse,
    summary="Liveness & readiness check",
    tags=["System"],
)
async def health():
    """
    Returns the status of each dependency:
    - **neo4j**: connectivity to Neo4j Aura
    - **ollama**: connectivity to Ollama + model availability
    """
    neo4j_status = "connected"
    ollama_status = "ready"
    overall = "ok"

    # Check Neo4j
    try:
        pipe = get_pipeline()
        pipe._run_cypher("RETURN 1 AS ping")
    except Exception as e:
        neo4j_status = f"error: {e}"
        overall = "degraded"

    # Check Ollama
    try:
        r = requests.get(f"{cfg.ollama_host}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(cfg.llm_model in m for m in models):
            ollama_status = f"model '{cfg.llm_model}' not pulled"
            overall = "degraded"
    except Exception as e:
        ollama_status = f"error: {e}"
        overall = "degraded"

    return HealthResponse(
        status=overall,
        neo4j=neo4j_status,
        ollama=ollama_status,
        llm_model=cfg.llm_model,
        embed_model=cfg.embed_model_name,
    )


@app.post(
    "/api/v1/ask",
    response_model=AskResponse,
    summary="Ask a legal question (full RAG)",
    tags=["RAG"],
)
async def ask(req: AskRequest):
    """
    **Full RAG pipeline:**

    1. LLM extracts keywords, legal topics, and article numbers from the question
    2. Hybrid retrieval: keyword graph search + section search + semantic vector search
    3. Reranking: 55% semantic similarity · 35% keyword overlap · 10% direct-article bonus
    4. Qwen3 generates an answer citing the retrieved articles

    Supports questions in **English** and **Arabic**.
    """
    pipe = get_pipeline()
    result = await pipe.ask(req.question, top_k=req.top_k)

    articles = [
        ArticleOut(
            id=a["id"],
            number=a["number"],
            english=a["english"] or "",
            arabic=a["arabic"] or "",
            score=round(a.get("score", 0.0), 4),
        )
        for a in result["articles"]
    ]

    raw_meta = result.get("metadata", {})
    metadata = MetadataOut(
        keywords_en=raw_meta.get("keywords_en", []),
        keywords_ar=raw_meta.get("keywords_ar", []),
        legal_topics=raw_meta.get("legal_topics", []),
        article_numbers=raw_meta.get("article_numbers", []),
        search_query=raw_meta.get("search_query", ""),
    )

    settings = get_settings()
    return AskResponse(
        answer=result["answer"],
        articles=articles,
        metadata=metadata,
        processing_time_ms=result["processing_time_ms"],
        generation=GenerationInfo(
            model=settings.llm_model,
            params={
                "runtime": "ollama",
                "temperature": settings.llm_temp_answer,
                "temperature_extract": settings.llm_temp_extract,
            },
            embed_model=settings.embed_model_name,
        ),
    )


@app.post("/api/v1/ask/stream", summary="Ask (full RAG) — token-streamed (NDJSON)", tags=["RAG"])
async def ask_stream(req: AskRequest):
    """Same RAG pipeline as /ask, but streams the answer token-by-token as
    newline-delimited JSON: a `meta` line (articles + metadata + generation),
    then `delta` lines, then a `done` line."""
    pipe = get_pipeline()
    settings = get_settings()
    gen_info = {
        "model": settings.llm_model,
        "params": {"runtime": "ollama", "temperature": settings.llm_temp_answer,
                   "temperature_extract": settings.llm_temp_extract},
        "embed_model": settings.embed_model_name,
    }

    def _events():
        try:
            for ev in pipe.ask_stream(req.question, top_k=req.top_k):
                if ev.get("type") == "meta":
                    ev["articles"] = [
                        {"id": a.get("id"), "number": a.get("number"),
                         "english": a.get("english") or "", "arabic": a.get("arabic") or "",
                         "score": round(a.get("score", 0.0), 4)}
                        for a in ev.get("articles", [])
                    ]
                    ev["generation"] = gen_info
                yield json.dumps(ev, ensure_ascii=False) + "\n"
        except Exception as e:  # surface as a terminal error event
            yield json.dumps({"type": "error", "detail": f"{type(e).__name__}: {e}"}) + "\n"

    return StreamingResponse(_events(), media_type="application/x-ndjson")


@app.post(
    "/api/v1/search",
    response_model=list[ArticleOut],
    summary="Hybrid article search (no LLM answer)",
    tags=["Retrieval"],
)
async def search(req: SearchRequest):
    """
    Runs the retrieval pipeline without answer generation.
    Useful for building custom UIs or chaining with other services.

    Returns a ranked list of matching articles.
    """
    pipe = get_pipeline()
    articles = await pipe.search(req.query, top_k=req.top_k)
    return [
        ArticleOut(
            id=a["id"],
            number=a["number"],
            english=a["english"] or "",
            arabic=a["arabic"] or "",
            score=round(a.get("score", 0.0), 4),
        )
        for a in articles
    ]


@app.get(
    "/api/v1/article/{number}",
    response_model=ArticleOut,
    summary="Fetch a single article by its number",
    tags=["Retrieval"],
)
async def get_article(number: int):
    """
    Returns the Arabic and English text of a specific article.

    Example: `/api/v1/article/5` returns Article 5 of the Egyptian Civil Law.
    """
    if number < 1:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Article number must be a positive integer.",
        )
    pipe   = get_pipeline()
    result = await asyncio.to_thread(pipe.get_article, number)  # type: ignore[arg-type]
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Article {number} not found.",
        )
    return ArticleOut(
        id=result["id"],
        number=result["number"],
        english=result["english"] or "",
        arabic=result["arabic"] or "",
        score=0.0,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Dev entry point
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import asyncio
    import uvicorn

    uvicorn.run(
        "apps.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=True,
    )
