"""
Bilingual RAG API (Project 4) — mirrors apps/api shape so the unified UI can
call either RAG service over HTTP with the same client code.

Run:
    uvicorn apps.bilingual_rag.api:app --host 0.0.0.0 --port 8100

Endpoints:
    GET  /health        → liveness + readiness
    POST /api/v1/ask    → full RAG (answer + retrieved chunks)
    POST /api/v1/search → retrieval only (no LLM answer)
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import requests
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import BilingualRAGPipeline

_pipeline: BilingualRAGPipeline | None = None
cfg = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = BilingualRAGPipeline(cfg)
    # Touch the collection early so a missing index fails fast & loudly.
    _ = _pipeline.collection
    print("[brag-api] ready.")
    yield


def get_pipeline() -> BilingualRAGPipeline:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — server still starting up.")
    return _pipeline


app = FastAPI(
    title="Egyptian Civil Code — Bilingual RAG API",
    description="BGE-M3 + Chroma + multilingual cross-encoder rerank + Qwen (Ollama).",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)
    use_rerank: bool = True


class Hit(BaseModel):
    article_number: int
    language: str
    score: float = 0.0
    rerank_score: float | None = None
    section_path: str = ""
    text: str


class AskResponse(BaseModel):
    answer: str
    keywords: list[str] | None = None
    article_numbers: list[int] = []
    search_query: str = ""
    detected_language: str | None = None
    hits: list[Hit]
    processing_time_ms: int = 0


def _to_hits(raw: list[dict]) -> list[Hit]:
    return [
        Hit(
            article_number=h["article_number"], language=h["language"],
            score=round(float(h.get("score", 0.0)), 4),
            rerank_score=(round(float(h["rerank_score"]), 4) if "rerank_score" in h else None),
            section_path=h.get("section_path", ""), text=h["text"],
        )
        for h in raw
    ]


@app.get("/health", tags=["System"])
async def health():
    overall, ollama_status = "ok", "ready"
    try:
        _ = get_pipeline().collection.count()
        chroma_status = "connected"
    except Exception as e:
        chroma_status, overall = f"error: {e}", "degraded"
    try:
        r = requests.get(f"{cfg.ollama_host}/api/tags", timeout=5)
        models = [m["name"] for m in r.json().get("models", [])]
        if not any(cfg.llm_model in m for m in models):
            ollama_status, overall = f"model '{cfg.llm_model}' not pulled", "degraded"
    except Exception as e:
        ollama_status, overall = f"error: {e}", "degraded"
    return {
        "status": overall, "chroma": chroma_status, "ollama": ollama_status,
        "llm_model": cfg.llm_model, "embed_model": cfg.embed_model_name,
    }


@app.post("/api/v1/ask", response_model=AskResponse, tags=["RAG"])
async def ask(req: AskRequest):
    import asyncio
    pipe = get_pipeline()
    r = await asyncio.to_thread(pipe.answer, req.question, k=req.top_k, use_rerank=req.use_rerank)
    return AskResponse(
        answer=r["answer"], keywords=r.get("keywords"),
        article_numbers=r.get("article_numbers") or [],
        search_query=r.get("search_query", ""), detected_language=r.get("detected_language"),
        hits=_to_hits(r["hits"]), processing_time_ms=r.get("processing_time_ms", 0),
    )


@app.post("/api/v1/search", response_model=list[Hit], tags=["Retrieval"])
async def search(req: AskRequest):
    import asyncio
    pipe = get_pipeline()
    r = await asyncio.to_thread(pipe.retrieve, req.question, k=req.top_k, use_rerank=req.use_rerank)
    return _to_hits(r["hits"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.bilingual_rag.api:app", host=cfg.api_host, port=cfg.api_port, reload=False)
