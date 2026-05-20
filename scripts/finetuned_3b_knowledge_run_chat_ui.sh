#!/usr/bin/env bash
# Launch the Legal Policy chat UI (Stage A v2 — closed-book Qwen2.5-3B knowledge adapter).
#
# Uses the `legalpolicy` conda env, which has gradio + the CUDA model stack
# (torch+cu121, peft, transformers, bitsandbytes). Override the interpreter with
# $LP_PYTHON if your env lives elsewhere.
#
#   bash scripts/finetuned_3b_knowledge_run_chat_ui.sh            # serve on http://localhost:7860
#   LP_SHARE=1 bash scripts/finetuned_3b_knowledge_run_chat_ui.sh # also create a public share link
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${LP_PYTHON:-/home/aya/miniconda3/envs/legalpolicy/bin/python}"
exec "$PY" -m app.finetuned_3b_knowledge_chat_app
