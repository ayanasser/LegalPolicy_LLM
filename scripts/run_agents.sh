#!/usr/bin/env bash
# Project 5 — Multi-agent LangGraph orchestrator (Gradio, port 7860).
# Needs ANTHROPIC_BASE_URL + ANTHROPIC_AUTH_TOKEN (glm-5 gateway) in .env,
# claude-agent-sdk + lightrag-hku installed, and Ollama qwen3-embedding:0.6b
# for the LightRAG retrieval tool.
source "$(dirname "${BASH_SOURCE[0]}")/_env.sh"
ensure_ollama
ollama list | grep -q "qwen3-embedding:0.6b" || ollama pull qwen3-embedding:0.6b
exec "$LP_PY" -m legal_explainer.agents.gradio_app
