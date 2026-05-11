"""Generate predictions from the QLoRA adapter on a stratified val sample.

Output: reports/eval/judge_predictions.json — list of records each containing
{id, language, kind, article_key, prompt, reference, prediction, gen_seconds}.

A separate LLM-as-judge step reads this file and scores each prediction.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "runs" / "qlora-qwen2.5-1.5b-v1"
DEFAULT_VAL_PATH = PROJECT_ROOT / "data" / "qa_pairs_val.baseline.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "reports" / "eval" / "judge_predictions.json"


def stratified_sample(rows, n_en=10, n_ar=10, seed=13):
    rng = random.Random(seed)
    en = [r for r in rows if r["language"] == "en" and r["kind"] == "explanation"]
    ar = [r for r in rows if r["language"] == "ar" and r["kind"] == "explanation"]
    refusals = [r for r in rows if r["kind"] == "refusal"]
    rng.shuffle(en)
    rng.shuffle(ar)
    return en[:n_en] + ar[:n_ar] + refusals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-en", type=int, default=10)
    ap.add_argument("--n-ar", type=int, default=10)
    ap.add_argument("--max-new-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR,
                    help="Path to the LoRA adapter directory.")
    ap.add_argument("--val-path", type=Path, default=DEFAULT_VAL_PATH,
                    help="JSONL of validation examples to sample from.")
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT_PATH,
                    help="Where to write the predictions JSON.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.val_path.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(rows, n_en=args.n_en, n_ar=args.n_ar)
    print(f"Sampled {len(sample)} examples (EN={args.n_en}, AR={args.n_ar}, "
          f"refusals={len([r for r in sample if r['kind']=='refusal'])})")
    print(f"Adapter: {args.adapter_dir}")
    print(f"Val:     {args.val_path}")
    print(f"Out:     {args.out_path}")

    print("Loading tokenizer and adapter (4-bit) ...")
    tok = AutoTokenizer.from_pretrained(str(args.adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(args.adapter_dir),
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    print("Model loaded. Beginning generation ...")

    args.out_path.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for i, ex in enumerate(sample, 1):
        user_msg = ex["messages"][0]["content"]
        reference = ex["messages"][1]["content"]
        chat = [{"role": "user", "content": user_msg}]
        prompt_text = tok.apply_chat_template(
            chat, tokenize=False, add_generation_prompt=True
        )
        inputs = tok(prompt_text, return_tensors="pt").to(model.device)
        prompt_len = inputs["input_ids"].shape[1]
        t0 = time.perf_counter()
        with torch.no_grad():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tok.eos_token_id,
                repetition_penalty=1.05,
            )
        elapsed = time.perf_counter() - t0
        new_tokens = generated[0, prompt_len:]
        prediction = tok.decode(new_tokens, skip_special_tokens=True).strip()

        rec = {
            "id": i,
            "language": ex["language"],
            "kind": ex["kind"],
            "article_key": ex.get("article_key"),
            "prompt": user_msg,
            "reference": reference,
            "prediction": prediction,
            "gen_seconds": round(elapsed, 2),
        }
        out.append(rec)
        print(f"[{i}/{len(sample)}] lang={ex['language']} kind={ex['kind']} "
              f"art={ex.get('article_key')} time={elapsed:.1f}s "
              f"chars={len(prediction)}")

        # Save incrementally so we don't lose progress.
        args.out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nWrote {len(out)} predictions -> {args.out_path}")


if __name__ == "__main__":
    main()
