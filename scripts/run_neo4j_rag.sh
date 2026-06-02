#!/usr/bin/env bash
# Project 3 — Neo4j Graph RAG API (FastAPI, port 8000).
# Requires NEO4J_* in .env and Ollama qwen3:4b. BGE-M3 runs via FlagEmbedding.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama
ollama list | grep -q "qwen3:4b" || ollama pull qwen3:4b
# Make sure bge-m3 has a safetensors variant (torch<2.6 + transformers 5.x).
"$LP_PY" scripts/ensure_bge_m3_safetensors.py || true
exec "$LP_PY" -m uvicorn apps.api.main:app --host 0.0.0.0 --port "${LP_NEO4J_PORT:-8000}"
