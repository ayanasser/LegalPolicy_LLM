#!/usr/bin/env bash
# Project 1 — Prompt Design CLI assistant (Ollama llama3.2:3b).
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama
ollama list | grep -q "llama3.2:3b" || ollama pull llama3.2:3b
exec "$LP_PY" "src/Prompt Design/legal_policy_assistant_egypt_v2.py"
