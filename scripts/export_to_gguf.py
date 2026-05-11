"""Merge a LoRA adapter and export to GGUF (q4_K_M by default) for Ollama.

Requires a local llama.cpp checkout that has been built with the
`llama-quantize` binary, and whose `convert_hf_to_gguf.py` is present.

Usage:
    python scripts/export_to_gguf.py \\
        --adapter   runs/qlora-qwen2.5-1.5b-v1 \\
        --base      Qwen/Qwen2.5-1.5B-Instruct \\
        --llama-cpp ~/llama.cpp \\
        --out-name  legalpolicy-qwen1.5b
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import torch  # noqa: E402

from legal_explainer.finetune.merge_export import merge  # noqa: E402

OLLAMA_MODELFILE_TEMPLATE = '''FROM ./{gguf_filename}
TEMPLATE """<|im_start|>system
{{{{ .System }}}}<|im_end|>
<|im_start|>user
{{{{ .Prompt }}}}<|im_end|>
<|im_start|>assistant
"""
PARAMETER stop "<|im_end|>"
PARAMETER temperature 0.3
SYSTEM "You are a careful explainer of the Egyptian Civil Code. Provide plain-language explanations grounded in the cited articles. Always include a one-line disclaimer that you are not providing legal advice."
'''


def run(cmd: list):
    print("$", " ".join(str(c) for c in cmd))
    subprocess.run([str(c) for c in cmd], check=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--base", required=True, type=str)
    ap.add_argument("--llama-cpp", required=True, type=Path,
                    help="Path to a built llama.cpp checkout.")
    ap.add_argument("--out-dir", default=PROJECT_ROOT / "artifacts", type=Path)
    ap.add_argument("--out-name", default="legalpolicy-qwen", type=str)
    ap.add_argument("--quant", default="q4_K_M",
                    choices=["q4_K_M", "q5_K_M", "q8_0", "f16"])
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    merged_dir = args.out_dir / f"{args.out_name}_merged"
    fp16_gguf = args.out_dir / f"{args.out_name}_fp16.gguf"
    quant_gguf = args.out_dir / f"{args.out_name}_{args.quant}.gguf"
    modelfile = args.out_dir / f"{args.out_name}.Modelfile"

    print("=== Step 1: merge LoRA into base ===")
    merge(args.base, args.adapter, merged_dir, dtype=torch.bfloat16)

    print("=== Step 2: convert merged HF model to GGUF (fp16) ===")
    convert_script = args.llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(f"Not found: {convert_script}")
    run([sys.executable, convert_script, merged_dir,
         "--outfile", fp16_gguf, "--outtype", "f16"])

    print(f"=== Step 3: quantize to {args.quant} ===")
    quantize_bin = args.llama_cpp / "build" / "bin" / "llama-quantize"
    if not quantize_bin.exists():
        quantize_bin = args.llama_cpp / "llama-quantize"
    if not quantize_bin.exists():
        raise FileNotFoundError(
            f"llama-quantize not found at {args.llama_cpp}. "
            "Build llama.cpp first: cd llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j"
        )
    run([quantize_bin, fp16_gguf, quant_gguf, args.quant])

    print("=== Step 4: write Ollama Modelfile ===")
    modelfile.write_text(
        OLLAMA_MODELFILE_TEMPLATE.format(gguf_filename=quant_gguf.name),
        encoding="utf-8",
    )

    print(f"\nDone.\n  Merged HF: {merged_dir}\n  GGUF:      {quant_gguf}\n  Modelfile: {modelfile}")
    print("\nNext step:")
    print(f"  cd {args.out_dir}")
    print(f"  ollama create {args.out_name} -f {modelfile.name}")
    print(f"  ollama run {args.out_name}")


if __name__ == "__main__":
    main()
