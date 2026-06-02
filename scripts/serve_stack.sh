#!/usr/bin/env bash
# Bring up the full unified-UI stack:
#   - Neo4j Graph RAG API     on :8000   (needs NEO4J_* in .env)
#   - Bilingual RAG API       on :8100   (needs the Chroma index built)
#   - Unified UI              on :7870
#
# RAG services run in the background; the UI runs in the foreground. Ctrl-C
# stops the UI and tears the services down. Logs go to /tmp/lp_*.log.
#
# VRAM note (6 GB laptop): the two RAG services load BGE-M3 on CPU by default
# (BRAG_EMBED_DEVICE=cpu), leaving the GPU for the local Qwen the UI loads when
# you pick a baseline/finetuned backend. Test one heavy backend at a time.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama

PIDS=()
cleanup() { echo; echo "[stack] stopping services …"; for p in "${PIDS[@]}"; do kill "$p" 2>/dev/null || true; done; }
trap cleanup EXIT INT TERM

echo "[stack] starting Bilingual RAG API on :8100 → /tmp/lp_bilingual.log"
( ollama list | grep -q "qwen2.5:3b-instruct" || ollama pull qwen2.5:3b-instruct ) >/dev/null 2>&1 || true
"$LP_PY" -m uvicorn apps.bilingual_rag.api:app --host 0.0.0.0 --port 8100 >/tmp/lp_bilingual.log 2>&1 &
PIDS+=($!)

if grep -q "^NEO4J_PASSWORD=..*" .env 2>/dev/null; then
  echo "[stack] starting Neo4j Graph RAG API on :8000 → /tmp/lp_neo4j.log"
  ( ollama list | grep -q "qwen3:4b" || ollama pull qwen3:4b ) >/dev/null 2>&1 || true
  "$LP_PY" scripts/ensure_bge_m3_safetensors.py >/dev/null 2>&1 || true
  "$LP_PY" -m uvicorn apps.api.main:app --host 0.0.0.0 --port 8000 >/tmp/lp_neo4j.log 2>&1 &
  PIDS+=($!)
else
  echo "[stack] NEO4J_PASSWORD not set in .env — skipping Neo4j RAG service."
fi

# Project 6 (Combined: legal prompt + graph RAG → finetuned answer) on :8200.
# Off by default (it loads BGE-M3 + the finetuned Qwen on the GPU). Enable with
# LP_WITH_LGF=1, and only when the GPU is free.
if [ "${LP_WITH_LGF:-0}" = "1" ] && grep -q "^NEO4J_PASSWORD=..*" .env 2>/dev/null; then
  echo "[stack] starting Combined (Project 6) API on :8200 → /tmp/lp_lgf.log"
  "$LP_PY" -m uvicorn apps.legal_graphrag_finetuned.api:app --host 0.0.0.0 --port 8200 >/tmp/lp_lgf.log 2>&1 &
  PIDS+=($!)
fi

echo "[stack] starting Unified UI on :7870 (Ctrl-C to stop everything)"
"$LP_PY" -m apps.unified_ui.app
