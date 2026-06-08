"""
Configuration for the Bilingual RAG service (Project 4).

Converted from notebooks/bilingual-rag-system-over-the-egyptian-civil-code_fixed_trial.ipynb.
Everything is overridable via environment variables so the same code runs the
standalone Gradio app, the FastAPI service, and the index builder.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class BRagSettings(BaseSettings):
    # ── Corpus ───────────────────────────────────────────────────────────────
    corpus_path: str = Field(str(PROJECT_ROOT / "data" / "orig_data.json"), alias="BRAG_CORPUS")

    # ── Embeddings (BGE-M3 via sentence-transformers) ────────────────────────
    embed_model_name: str = Field("BAAI/bge-m3", alias="BRAG_EMBED_MODEL")
    # Default to CPU so the index build / queries don't fight the local GPU
    # (e.g. while a QLoRA run is training). Set BRAG_EMBED_DEVICE=cuda to use GPU.
    embed_device: str = Field("cpu", alias="BRAG_EMBED_DEVICE")

    # ── Chunking (characters) ────────────────────────────────────────────────
    chunk_size: int = 800
    chunk_overlap: int = 120

    # ── Chroma persistent store ──────────────────────────────────────────────
    chroma_dir: str = Field(
        str(PROJECT_ROOT / "artifacts" / "bilingual_rag" / "chroma_db"),
        alias="BRAG_CHROMA_DIR",
    )
    collection_name: str = "egyptian_civil_code"
    insert_batch_size: int = 64

    # ── Reranker (multilingual cross-encoder) ────────────────────────────────
    use_reranker: bool = Field(True, alias="BRAG_USE_RERANKER")
    reranker_name: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    # Reranker device — falls back to embed_device when empty, so a single
    # BRAG_EMBED_DEVICE=cuda runs BOTH the embedder and the reranker on the GPU
    # (big speedup on Colab/A100 vs the CPU default).
    rerank_device: str = Field("", alias="BRAG_RERANK_DEVICE")

    # ── Retrieval defaults ───────────────────────────────────────────────────
    top_k: int = 5
    candidate_k: int = 20
    restrict_language: bool = True
    use_keywords: bool = Field(True, alias="BRAG_USE_KEYWORDS")

    # ── LLM (Ollama Qwen 3B) ─────────────────────────────────────────────────
    ollama_host: str = Field("http://localhost:11434", alias="OLLAMA_HOST")
    llm_model: str = Field("qwen2.5:3b-instruct", alias="BRAG_OLLAMA_MODEL")
    llm_temperature: float = 0.0
    llm_num_predict: int = 512

    # ── API server ───────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8100

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        populate_by_name=True,
        extra="ignore",
    )


@lru_cache
def get_settings() -> BRagSettings:
    return BRagSettings()
