"""
Ensure a local `model.safetensors` exists for BAAI/bge-m3 (and any other
.bin-only embedding model), so it loads under transformers 5.x + torch < 2.6.

Why: BAAI/bge-m3 publishes only `pytorch_model.bin`. transformers 5.x refuses
`torch.load` on torch < 2.6 (CVE-2025-32434) unless the weights are safetensors.
Both RAG services (apps/bilingual_rag — SentenceTransformer; apps/api —
FlagEmbedding) hit this. Our own torch.load is fine, so we convert once and drop
a safetensors file into the model's HF cache snapshot.

Run (after the model is cached, i.e. downloaded at least once):
    python scripts/ensure_bge_m3_safetensors.py
    python scripts/ensure_bge_m3_safetensors.py BAAI/bge-m3 some/other-model
"""
from __future__ import annotations

import sys

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import save_file


def convert(repo: str) -> None:
    snap = snapshot_download(repo, local_files_only=True)
    import os
    bin_path = os.path.join(snap, "pytorch_model.bin")
    st_path = os.path.join(snap, "model.safetensors")
    if os.path.exists(st_path):
        print(f"[ok] {repo}: model.safetensors already present")
        return
    if not os.path.exists(bin_path):
        print(f"[skip] {repo}: no pytorch_model.bin in {snap}")
        return
    print(f"[convert] {repo}: {bin_path} → model.safetensors")
    sd = torch.load(bin_path, map_location="cpu", weights_only=True)
    sd = {k: v.clone().contiguous() for k, v in sd.items() if isinstance(v, torch.Tensor)}
    save_file(sd, st_path, metadata={"format": "pt"})
    print(f"[done] {repo}: wrote {st_path} ({os.path.getsize(st_path) / 1e6:.0f} MB)")


if __name__ == "__main__":
    repos = sys.argv[1:] or ["BAAI/bge-m3"]
    for r in repos:
        try:
            convert(r)
        except Exception as e:
            print(f"[error] {r}: {type(e).__name__}: {e}")
