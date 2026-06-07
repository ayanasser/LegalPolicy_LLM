#!/usr/bin/env bash
# Project 4 — Bilingual RAG.
#   ./scripts/run_bilingual_rag.sh build   → build the Chroma index (one-time)
#   ./scripts/run_bilingual_rag.sh ui      → standalone Gradio app (port 7861)
#   ./scripts/run_bilingual_rag.sh api     → FastAPI service (port 8100)
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
cmd="${1:-ui}"
case "$cmd" in
  build)
    "$LP_PY" scripts/ensure_bge_m3_safetensors.py || true
    exec "$LP_PY" -m apps.bilingual_rag.build_index ;;
  ui)
    ensure_ollama
    ollama list | grep -q "qwen2.5:3b-instruct" || ollama pull qwen2.5:3b-instruct
    exec "$LP_PY" -m apps.bilingual_rag.gradio_app ;;
  api)
    ensure_ollama
    ollama list | grep -q "qwen2.5:3b-instruct" || ollama pull qwen2.5:3b-instruct
    exec "$LP_PY" -m uvicorn apps.bilingual_rag.api:app --host 0.0.0.0 --port "${BRAG_API_PORT:-8100}" ;;
  *)
    echo "usage: $0 {build|ui|api}"; exit 1 ;;
esac
