#!/usr/bin/env bash
# Shared helpers for the run scripts. Source this at the top of each script.
#
#   LP_PY  — python interpreter to use (default: the legalpolicy conda env).
#   Run all scripts from the project root.
set -euo pipefail

# Resolve project root = parent of this scripts/ dir.
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

# Default to the legalpolicy conda env's python; override with LP_PY=...
LP_PY="${LP_PY:-/home/aya/miniconda3/envs/legalpolicy/bin/python}"
export PYTHONPATH="${ROOT}/src:${PYTHONPATH:-}"

ollama_up() { curl -s "${OLLAMA_HOST:-http://localhost:11434}/api/tags" >/dev/null 2>&1; }

ensure_ollama() {
  if ! ollama_up; then
    echo "[run] starting ollama serve …"
    nohup ollama serve >/tmp/ollama.log 2>&1 &
    for _ in $(seq 1 30); do ollama_up && break; sleep 1; done
  fi
  ollama_up && echo "[run] ollama is up." || { echo "[run] ERROR: ollama not reachable"; exit 1; }
}
