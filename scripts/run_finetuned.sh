#!/usr/bin/env bash
# Project 2 — Finetuned Qwen2.5-3B knowledge adapter chat (Gradio, port 7860).
# Needs free GPU (~2.5 GB). Don't run while a QLoRA training job owns the card.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
exec "$LP_PY" -m app.finetuned_3b_knowledge_chat_app
