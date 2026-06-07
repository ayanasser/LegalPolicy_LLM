#!/usr/bin/env bash
# Project 6 — Legal Graph-RAG + Finetuned model (standalone Gradio, port 7863).
#   legal prompt + safety → Neo4j graph retrieval → finetuned 3B answer.
# Self-contained: loads BGE-M3 retrieval + the finetuned Qwen in ONE process.
# Needs NEO4J_* in .env, Ollama qwen3:4b, and a free GPU (~4-5 GB).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama
ollama list | grep -q "qwen3:4b" || ollama pull qwen3:4b
"$LP_PY" scripts/ensure_bge_m3_safetensors.py || true
exec "$LP_PY" -m apps.legal_graphrag_finetuned.app
