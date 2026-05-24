"""Shared configuration for the legal-explainer agents.

One source of truth for: paths, model selection, SDK auth env, and tunables.
Import this module instead of repeating these constants in every file.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path("/Volumes/Shared/NileUni/GenAI/LegalPolicy_LLM")
AGENTS_DIR = PROJECT_ROOT / "src" / "legal_explainer" / "agents"
DATA_DIR = AGENTS_DIR / "data"
PROMPTS_DIR = AGENTS_DIR / "prompts"
RAG_WORKING_DIR = AGENTS_DIR / "rag_storage_egyptian_law"

GLOSSARY_PATH = DATA_DIR / "glossary.json"
ARTICLES_INDEX_PATH = DATA_DIR / "articles_index.json"
PDF_PATH = PROJECT_ROOT / "EgyptianLaw.pdf"

EVAL_DATA_DIR = PROJECT_ROOT / "data"
EVAL_OUTPUT_DIR = PROJECT_ROOT / "reports" / "agent_eval"
TRACE_DIR = PROJECT_ROOT / "reports" / "agent_traces"

# ── Auth ─────────────────────────────────────────────────────────────────────
load_dotenv(PROJECT_ROOT / ".env")

SDK_ENV = {
    "ANTHROPIC_AUTH_TOKEN": os.getenv("ANTHROPIC_AUTH_TOKEN", ""),
    "ANTHROPIC_BASE_URL": os.getenv("ANTHROPIC_BASE_URL", ""),
    "ANTHROPIC_API_KEY": os.getenv("ANTHROPIC_API_KEY", ""),
    "API_TIMEOUT_MS": "3000000",
}

# ── Models ───────────────────────────────────────────────────────────────────
# Models map to roles. Override any of these from the calling code if needed.
ORCHESTRATOR_MODEL = "glm-5"
RESEARCHER_MODEL = "glm-5"
EXPLAINER_MODEL = "glm-5"
COMPARATOR_MODEL = "glm-5"
ROUTER_MODEL = "glm-5"

# ── RAG tunables ─────────────────────────────────────────────────────────────
RAG_LLM_MODEL = ORCHESTRATOR_MODEL
RAG_DEFAULT_MODE = "hybrid"
RAG_DEFAULT_TOP_K = 5

# ── Ollama embeddings (must match what ingest_rag.ipynb used) ────────────────
OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_EMBED_MODEL = "qwen3-embedding:0.6b"
OLLAMA_EMBED_DIM = 1024
OLLAMA_EMBED_MAX_TOKENS = 32768

# ── Subagent temperatures (passed via system prompt — SDK doesn't expose temp) ─
# Kept here as documentation. If you migrate to a transport that supports it,
# wire them into llm_model_kwargs.
TEMP_RESEARCHER = 0.2
TEMP_EXPLAINER = 0.4
TEMP_COMPARATOR = 0.3