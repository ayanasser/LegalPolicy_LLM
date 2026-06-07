#!/usr/bin/env bash
# The Unified UI (Gradio, port 7870). One chat, dropdown picks the project.
# The RAG backends call the two FastAPI services over HTTP — start those
# separately (./scripts/serve_stack.sh) if you want to use them.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama
exec "$LP_PY" -m apps.unified_ui.app
