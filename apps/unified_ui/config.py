"""Configuration for the Unified UI.

One place to point the UI at the local models, the Ollama server, and the two
RAG microservices (Neo4j RAG on :8000, Bilingual RAG on :8100). Everything is
overridable via environment variables.
"""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ── Local HF model (shared base for baseline + finetuned adapter) ─────────────
BASE_QWEN_MODEL = os.getenv("LP_BASE_QWEN", "Qwen/Qwen2.5-3B-Instruct")
KNOWLEDGE_ADAPTER_DIR = os.getenv(
    "LP_ADAPTER_DIR", str(PROJECT_ROOT / "runs" / "qlora-qwen2.5-3b-knowledge")
)

# ── Ollama (prompt-design + llama baseline) ──────────────────────────────────
OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
LLAMA_BASELINE_MODEL = os.getenv("LP_LLAMA_MODEL", "llama3.2:3b")
PROMPT_DESIGN_MODEL = os.getenv("LP_PROMPT_MODEL", "llama3.2:3b")
PROMPT_DESIGN_QWEN_MODEL = os.getenv("LP_PROMPT_QWEN_MODEL", "qwen2.5:3b-instruct")

# ── RAG microservices (HTTP) ─────────────────────────────────────────────────
NEO4J_RAG_URL = os.getenv("LP_NEO4J_RAG_URL", "http://localhost:8000")
BILINGUAL_RAG_URL = os.getenv("LP_BILINGUAL_RAG_URL", "http://localhost:8100")
# Project 6 — legal prompt + graph RAG → finetuned answer (standalone service).
LGF_RAG_URL = os.getenv("LP_LGF_RAG_URL", "http://localhost:8200")
RAG_HTTP_TIMEOUT = float(os.getenv("LP_RAG_TIMEOUT", "180"))

# ── Generation defaults for the local HF models ──────────────────────────────
MAX_NEW_TOKENS = int(os.getenv("LP_MAX_NEW_TOKENS", "600"))

# Preload the local HF model (baseline + finetuned share one 4-bit load) and run
# a tiny warm-up generation at UI startup, so the ~175s load + first-call CUDA
# warm-up are paid once at launch instead of on the user's first message. Off by
# default so RAG-only sessions start instantly; enable with LP_PRELOAD=1.
PRELOAD_LOCAL_MODEL = os.getenv("LP_PRELOAD", "0") == "1"

# ── Server ───────────────────────────────────────────────────────────────────
UI_PORT = int(os.getenv("LP_UI_PORT", "7870"))
UI_SHARE = os.getenv("LP_SHARE", "0") == "1"

# Data sources for suggested questions.
DATA_DIR = PROJECT_ROOT / "data"
GENERAL_CSV = DATA_DIR / "general_user_legal_questions.csv"
LAWYER_CSV = DATA_DIR / "lawyer_llm_solution_questions.csv"
KNOWLEDGE_JSONL = DATA_DIR / "qa_pairs_knowledge.jsonl"
