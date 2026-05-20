"""Interactive chat REPL for a trained QLoRA adapter.

Loads the adapter in 4-bit (fits the 6 GB laptop for the 1.5B / 3B bases) and
gives a terminal chat loop. Defaults to the Stage A v2 knowledge adapter.

Usage:
    python scripts/chat.py
    python scripts/chat.py --adapter-dir runs/qlora-qwen2.5-3b-combined
    python scripts/chat.py --temperature 0.0          # greedy (best for recall)
    python scripts/chat.py --no-history               # single-turn (recommended
                                                       #   for the knowledge adapter)

Commands inside the chat:
    /reset   clear the conversation history
    /quit    exit
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = PROJECT_ROOT / "runs" / "qlora-qwen2.5-3b-knowledge"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0.0 = greedy (best for verbatim recall). >0 = sampling.")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-history", action="store_true",
                    help="Single-turn: don't feed prior turns back in. Recommended "
                         "for the knowledge adapter (it isn't a conversational model).")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    print(f"Loading adapter: {args.adapter_dir}")
    tok = AutoTokenizer.from_pretrained(str(args.adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(args.adapter_dir), quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    p = torch.cuda.get_device_properties(0)
    print(f"Ready on {p.name} ({p.total_memory/1e9:.1f} GB). "
          f"Type a message, /reset to clear, /quit to exit.\n")

    history: list[dict] = []
    while True:
        try:
            user = input("\033[1myou>\033[0m ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nbye.")
            break
        if not user:
            continue
        if user in ("/quit", "/exit"):
            break
        if user == "/reset":
            history = []
            print("(history cleared)\n")
            continue

        messages = ([] if args.no_history else list(history)) + [{"role": "user", "content": user}]
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p if args.temperature > 0 else None,
                repetition_penalty=args.repetition_penalty,
                pad_token_id=tok.eos_token_id,
            )
        reply = tok.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        print(f"\033[1mmodel>\033[0m {reply}\n")
        if not args.no_history:
            history.append({"role": "user", "content": user})
            history.append({"role": "assistant", "content": reply})


if __name__ == "__main__":
    main()
