#!/usr/bin/env bash
# One-shot Colab (T4) setup for the unified-UI stack. Run ONCE per session,
# from the repo root, inside a Colab terminal:
#
#     bash scripts/colab_setup.sh
#
# It is idempotent — safe to re-run. It does NOT clone the repo, mount Drive,
# copy the adapter, or write .env (those are interactive / notebook steps —
# see the guide). After it finishes, start everything with:
#
#     LP_PY=$(which python) LP_SHARE=1 ./scripts/serve_stack.sh
#
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> 1/4  Python deps (torch is preinstalled on Colab — not touched)"
# Core stack: local HF Qwen + Bilingual RAG + Neo4j-Aura graph RAG.
pip install -q \
  gradio pandas python-dotenv pyyaml requests \
  transformers peft bitsandbytes accelerate sentencepiece einops \
  chromadb sentence-transformers ollama \
  fastapi 'uvicorn[standard]' pydantic pydantic-settings \
  neo4j FlagEmbedding langfuse
# Uncomment for the Multi-Agent (LangGraph) project too:
# pip install -q langchain langchain-community langgraph instructor \
#   claude-agent-sdk lightrag-hku nltk httpx duckduckgo-search

echo "==> 2/4  Ollama (install if missing, start, pull models)"
command -v ollama >/dev/null 2>&1 || curl -fsSL https://ollama.com/install.sh | sh
curl -s http://localhost:11434/api/tags >/dev/null 2>&1 || {
  echo "    starting ollama serve …"
  nohup ollama serve >/tmp/ollama.log 2>&1 &
  for _ in $(seq 1 30); do curl -s http://localhost:11434/api/tags >/dev/null 2>&1 && break; sleep 1; done
}
for m in qwen2.5:3b-instruct qwen3:4b llama3.2:3b; do
  ollama list | grep -q "$m" || { echo "    pulling $m"; ollama pull "$m"; }
done

echo "==> 3/4  Chroma index for the Bilingual RAG"
if [ -f artifacts/bilingual_rag/chroma_db/chroma.sqlite3 ]; then
  echo "    index already present — skipping build."
else
  echo "    building (BGE-M3 on GPU, ~1-2 min) …"
  BRAG_EMBED_DEVICE=cuda python -m apps.bilingual_rag.build_index
fi

echo "==> 4/4  Sanity checks"
[ -f .env ] && grep -q '^NEO4J_PASSWORD=..' .env \
  && echo "    .env: NEO4J_PASSWORD set ✓" \
  || echo "    .env: NEO4J_PASSWORD NOT set — graph RAG will be skipped. (Create .env first.)"
[ -d runs/qlora-qwen2.5-3b-knowledge ] \
  && echo "    adapter: present ✓" \
  || echo "    adapter: missing — the Finetuned project won't load. (Copy it from Drive.)"

echo
echo "Setup done. Start the stack with:"
echo "    LP_PY=\$(which python) LP_SHARE=1 ./scripts/serve_stack.sh"
echo "It prints a public *.gradio.live link and runs in the foreground (Ctrl-C stops all)."
