"""
Build the persistent Chroma index for the bilingual RAG service.

One-time (or whenever the corpus changes). Embeds every article chunk with
BGE-M3 and writes a persistent Chroma store under artifacts/bilingual_rag/.

Run:
    python -m apps.bilingual_rag.build_index

Notes:
  * Embeddings default to CPU (BRAG_EMBED_DEVICE=cpu) so this won't fight a
    running GPU job. On CPU this takes a few minutes; set BRAG_EMBED_DEVICE=cuda
    when the GPU is free for a faster build.
  * BAAI/bge-m3 (~2.3 GB) downloads on first run if not already cached.
"""
from __future__ import annotations

from .config import get_settings
from .pipeline import BilingualRAGPipeline


def main() -> None:
    cfg = get_settings()
    print(f"[brag] corpus     : {cfg.corpus_path}")
    print(f"[brag] chroma_dir : {cfg.chroma_dir}")
    print(f"[brag] embed      : {cfg.embed_model_name} on {cfg.embed_device}")
    pipe = BilingualRAGPipeline(cfg)
    n = pipe.build_index()
    print(f"[brag] done — {n} chunks indexed.")


if __name__ == "__main__":
    main()
