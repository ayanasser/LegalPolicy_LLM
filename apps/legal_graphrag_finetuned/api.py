"""FastAPI for Project 6 — Legal Graph-RAG + Finetuned model.

Run:
    uvicorn apps.legal_graphrag_finetuned.api:app --host 0.0.0.0 --port 8200

Endpoints:
    GET  /health        → readiness
    POST /api/v1/ask    → legal prompt + graph retrieval → finetuned answer
"""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from .config import get_settings
from .pipeline import LegalGraphRagFinetuned

cfg = get_settings()
_pipeline: LegalGraphRagFinetuned | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = LegalGraphRagFinetuned(cfg)
    _pipeline.startup()
    yield
    _pipeline.shutdown()


def get_pipeline() -> LegalGraphRagFinetuned:
    if _pipeline is None:
        raise RuntimeError("Pipeline not initialised — server still starting up.")
    return _pipeline


app = FastAPI(
    title="Legal Graph-RAG + Finetuned API",
    description="Legal prompt + Neo4j graph retrieval → finetuned Qwen2.5-3B knowledge adapter.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True,
                   allow_methods=["*"], allow_headers=["*"])


@app.exception_handler(Exception)
async def _err(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=2000)
    top_k: int = Field(5, ge=1, le=20)


class ArticleOut(BaseModel):
    number: int
    english: str = ""
    arabic: str = ""
    score: float = 0.0


class GenerationInfo(BaseModel):
    """The finetuned answer model + its decoding params (for traces)."""
    model: str
    params: dict = {}


class AskResponse(BaseModel):
    answer: str
    refused: bool = False
    articles: list[ArticleOut] = []
    processing_time_ms: int = 0
    generation: GenerationInfo | None = None


@app.get("/health", tags=["System"])
async def health():
    try:
        get_pipeline()._rag._run_cypher("RETURN 1 AS ping")
        return {"status": "ok", "adapter": cfg.adapter_dir}
    except Exception as e:
        return {"status": "degraded", "detail": str(e)}


@app.post("/api/v1/ask", response_model=AskResponse, tags=["RAG"])
async def ask(req: AskRequest):
    r = await get_pipeline().answer(req.question, top_k=req.top_k)
    return AskResponse(
        answer=r["answer"], refused=r.get("refused", False),
        articles=[ArticleOut(number=a.get("number"), english=a.get("english") or "",
                             arabic=a.get("arabic") or "", score=round(a.get("score", 0.0), 4))
                  for a in r["articles"]],
        processing_time_ms=r.get("processing_time_ms", 0),
        generation=GenerationInfo(
            model=f"{cfg.base_model} + QLoRA knowledge adapter ({cfg.adapter_dir.split('/')[-1]})",
            params={
                "runtime": "transformers + PEFT",
                "decoding": "greedy",
                "do_sample": False,
                "temperature": 0.0,
                "max_new_tokens": cfg.max_new_tokens,
                "repetition_penalty": 1.0,
                "quantization": "4-bit nf4 (double-quant)",
            },
        ),
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("apps.legal_graphrag_finetuned.api:app", host=cfg.api_host, port=cfg.api_port, reload=False)
