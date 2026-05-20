"""Generate predictions from the QLoRA adapter on a stratified val sample.

Output: reports/eval/judge_predictions.json — list of records each containing
{id, language, kind, article_key, prompt, reference, prediction, gen_seconds}.

A separate LLM-as-judge step reads this file and scores each prediction.
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from legal_explainer.finetune.dataset_builder import (  # noqa: E402
    _clean_article_text, _format_context_block, _eligible_distractor_keys, article_label,
)

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "runs" / "qlora-qwen2.5-1.5b-v1"
DEFAULT_VAL_PATH = PROJECT_ROOT / "data" / "qa_pairs_val.baseline.jsonl"
DEFAULT_OUT_PATH = PROJECT_ROOT / "reports" / "eval" / "judge_predictions.json"
DEFAULT_CORPUS_PATH = PROJECT_ROOT / "data" / "orig_data.json"

_ARTICLE_NUM_RE = re.compile(r"(?:Article|المادة)\s+(\d+)")


def build_raft_prompt(user_msg: str, language: str, corpus: dict, mode: str,
                      n_distractors: int = 1, rng: random.Random | None = None) -> str:
    """Prepend a RAFT context block to a plain question, matching training format.

    mode: 'none'  -> return user_msg unchanged (closed-book)
          'oracle' -> context = the asked article only
          'oracle+distractor' -> context = asked article + n_distractors random others (shuffled)
    """
    if mode == "none":
        return user_msg
    rng = rng or random.Random(13)
    m = _ARTICLE_NUM_RE.search(user_msg)
    field = "english" if language == "en" else "arabic"
    entries: list[tuple[str, str]] = []
    if m:
        key = f"Article {m.group(1)}"
        if key in corpus and isinstance(corpus[key], dict) and (corpus[key].get(field) or "").strip():
            entries.append((article_label(key, language),
                            _clean_article_text(corpus[key][field])))
        if mode == "oracle+distractor":
            pool = [k for k in _eligible_distractor_keys(corpus) if k != key]
            for dk in rng.sample(pool, k=min(n_distractors, len(pool))):
                entries.append((article_label(dk, language),
                                _clean_article_text(corpus[dk][field])))
            rng.shuffle(entries)
    if not entries:  # refusal cases or article not found: empty-context marker
        marker = ("[No relevant Egyptian Civil Code articles for this request.]"
                  if language == "en"
                  else "[لا توجد مواد ذات صلة من القانون المدني المصري لهذا الطلب.]")
        ctx_label = "Context" if language == "en" else "السياق"
        entries = [(ctx_label, marker)]
    return _format_context_block(entries, language) + user_msg


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
    ap.add_argument("--raft-context", choices=["none", "oracle", "oracle+distractor"],
                    default="none",
                    help="Prepend a RAFT-style context block to each question. "
                         "'none' = closed-book (plain question). 'oracle' = the asked "
                         "article only. 'oracle+distractor' = asked article + 1 distractor "
                         "(matches RAFT training format).")
    ap.add_argument("--corpus-path", type=Path, default=DEFAULT_CORPUS_PATH,
                    help="Path to orig_data.json (used to build RAFT context blocks).")
    args = ap.parse_args()

    rows = [json.loads(l) for l in args.val_path.read_text(encoding="utf-8").splitlines()]
    sample = stratified_sample(rows, n_en=args.n_en, n_ar=args.n_ar)
    corpus = json.loads(args.corpus_path.read_text(encoding="utf-8")) if args.raft_context != "none" else {}
    ctx_rng = random.Random(13)
    print(f"Sampled {len(sample)} examples (EN={args.n_en}, AR={args.n_ar}, "
          f"refusals={len([r for r in sample if r['kind']=='refusal'])})")
    print(f"Adapter:      {args.adapter_dir}")
    print(f"Val:          {args.val_path}")
    print(f"Out:          {args.out_path}")
    print(f"RAFT context: {args.raft_context}")

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
        if args.raft_context != "none":
            prompt_msg = build_raft_prompt(
                user_msg, ex["language"], corpus, args.raft_context, n_distractors=1, rng=ctx_rng,
            )
        else:
            prompt_msg = user_msg
        chat = [{"role": "user", "content": prompt_msg}]
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
            "raft_context_mode": args.raft_context,
            "prompt_with_context": prompt_msg if args.raft_context != "none" else None,
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
