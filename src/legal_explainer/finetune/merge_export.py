"""Merge a trained LoRA adapter back into the base model weights.

Produces a standalone HuggingFace-format checkpoint at <out_dir> that contains
the merged weights plus the tokenizer. The merged checkpoint is what gets fed
to llama.cpp's convert_hf_to_gguf.py downstream.

Usage:
    python -m legal_explainer.finetune.merge_export \\
        --adapter runs/qlora-qwen2.5-1.5b-v1 \\
        --base    Qwen/Qwen2.5-1.5B-Instruct \\
        --out     artifacts/qlora-qwen2.5-1.5b-v1-merged
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


def merge(base_id: str, adapter_dir: Path, out_dir: Path, dtype=torch.bfloat16):
    print(f"Loading base {base_id} (dtype={dtype}) on CPU...")
    base = AutoModelForCausalLM.from_pretrained(
        base_id, torch_dtype=dtype, device_map="cpu", trust_remote_code=True,
    )
    print(f"Applying adapter from {adapter_dir}...")
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)

    tok = AutoTokenizer.from_pretrained(base_id, trust_remote_code=True)
    tok.save_pretrained(str(out_dir))
    print(f"Merged model saved to {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", required=True, type=str, help="HF base model id")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16"])
    args = ap.parse_args()

    merge(args.base, args.adapter, args.out, dtype=getattr(torch, args.dtype))


if __name__ == "__main__":
    main()
