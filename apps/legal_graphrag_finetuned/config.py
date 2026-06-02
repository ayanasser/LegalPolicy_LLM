"""Configuration for the Legal Graph-RAG + Finetuned project (Project 6).

The graph-retrieval side reuses the Neo4j/Ollama/embedding settings from
apps/api (single source of truth). This file adds the finetuned-generator
settings. Everything is overridable via environment variables.
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]


class LGFSettings:
    # ── Finetuned generator (HF + PEFT, on the GPU) ──────────────────────────
    base_model: str = os.getenv("LGF_BASE_MODEL", "Qwen/Qwen2.5-3B-Instruct")
    adapter_dir: str = os.getenv(
        "LGF_ADAPTER_DIR", str(PROJECT_ROOT / "runs" / "qlora-qwen2.5-3b-knowledge")
    )
    max_new_tokens: int = int(os.getenv("LGF_MAX_NEW_TOKENS", "600"))

    # ── Retrieval ────────────────────────────────────────────────────────────
    top_k: int = int(os.getenv("LGF_TOP_K", "5"))

    # ── Server ───────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = int(os.getenv("LGF_API_PORT", "8200"))
    ui_port: int = int(os.getenv("LP_PORT", "7863"))


@lru_cache
def get_settings() -> LGFSettings:
    return LGFSettings()
