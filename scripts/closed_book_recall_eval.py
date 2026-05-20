"""Held-out closed-book recall eval — the thesis-grade number.

Samples kn_verbatim + kn_reverse examples from qa_pairs_knowledge_val.jsonl
(never seen in training: held-out task instances over articles that DO appear
in train; this is the canonical "in-distribution articles, held-out task
instances" test for whether the adapter has internalised the corpus content).

Grades each prediction:
  - kn_verbatim: char-similarity + token-recall + exact match vs the gold text
  - kn_reverse:  exact match on the article number extracted from the prediction

Writes a markdown report (with per-suite aggregates and per-item rows) and a
JSON of raw per-prompt records, both under reports/eval/.

Default sample size is **100 verbatim + 100 reverse = 200 prompts**, balanced
EN/AR. Override with `--n-verbatim` / `--n-reverse`. On the RTX 3050 each model
run takes ~60-90 min; `--also-base` doubles that.

Usage:
    # Stage A v2 (3B) adapter vs Qwen 2.5 3B base
    python scripts/closed_book_recall_eval.py \
        --adapter-dir runs/qlora-qwen2.5-3b-knowledge \
        --base-model Qwen/Qwen2.5-3B-Instruct \
        --also-base \
        --out-path reports/eval/closed_book_recall_3b.md

    # Stage A v1 (1.5B) adapter vs Qwen 2.5 1.5B base
    python scripts/closed_book_recall_eval.py \
        --adapter-dir runs/qlora-qwen2.5-1.5b-knowledge-v1 \
        --base-model Qwen/Qwen2.5-1.5B-Instruct \
        --also-base \
        --out-path reports/eval/closed_book_recall_1_5b.md

    # quick smoke (smaller N)
    python scripts/closed_book_recall_eval.py --n-verbatim 20 --n-reverse 20
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
import time
from collections import Counter
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from legal_explainer.finetune.knowledge_builder import clean_text  # noqa: E402

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "runs" / "qlora-qwen2.5-3b-knowledge"
DEFAULT_BASE_MODEL  = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_VAL_JSONL   = PROJECT_ROOT / "data" / "qa_pairs_knowledge_val.jsonl"
DEFAULT_OUT         = PROJECT_ROOT / "reports" / "eval" / "closed_book_recall.md"


# ---------------------------------------------------------------------------
# scoring helpers (kept consistent with smoke_test_knowledge_closed_book.py)
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[،؛؟.,;:!?\"'()\[\]{}«»\-—–]")
_ARTICLE_NUM_RE = re.compile(r"(?:Article|article|المادة|المادّة)\s*[#:]?\s*([0-9]+)")


def _normalize(text: str, lang: str) -> str:
    t = clean_text(text)
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if lang == "en":
        t = t.lower()
    return t


def char_similarity(pred: str, gold: str, lang: str) -> float:
    p, g = _normalize(pred, lang), _normalize(gold, lang)
    if not p or not g:
        return 0.0
    return SequenceMatcher(None, p, g).ratio()


def token_recall(pred: str, gold: str, lang: str) -> float:
    """Fraction of distinct gold tokens (len>=4) appearing in the prediction."""
    p = set(_normalize(pred, lang).split())
    g_tokens = [w for w in _normalize(gold, lang).split() if len(w) >= 4]
    if not g_tokens:
        return 0.0
    return len(set(g_tokens) & p) / len(set(g_tokens))


def extract_article_number(text: str) -> int | None:
    m = _ARTICLE_NUM_RE.search(text or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# sampling
# ---------------------------------------------------------------------------

def stratified_sample(rows: list[dict], kind: str, n: int, seed: int) -> list[dict]:
    """Equal-balance EN and AR within `kind`, sampled deterministically."""
    en = [r for r in rows if r.get("kind") == kind and r.get("language") == "en"]
    ar = [r for r in rows if r.get("kind") == kind and r.get("language") == "ar"]
    rng = random.Random(seed)
    rng.shuffle(en); rng.shuffle(ar)
    per = n // 2
    return en[:per] + ar[:n - per]


# ---------------------------------------------------------------------------
# model loading
# ---------------------------------------------------------------------------

def _bnb_4bit() -> BitsAndBytesConfig:
    return BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )


def load_adapter(adapter_dir: Path):
    from peft import AutoPeftModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir), quantization_config=_bnb_4bit(),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_base(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=_bnb_4bit(),
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return tok, model


def generate(tok, model, user_msg: str, *, max_new_tokens: int) -> tuple[str, float]:
    chat = [{"role": "user", "content": user_msg}]
    prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt_text, return_tensors="pt").to(model.device)
    prompt_len = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=False, repetition_penalty=1.0,
            pad_token_id=tok.eos_token_id,
        )
    sec = time.perf_counter() - t0
    pred = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
    return pred, sec


# ---------------------------------------------------------------------------
# suite runner
# ---------------------------------------------------------------------------

def run_suite(tok, model, label: str, verbatims: list[dict], reverses: list[dict],
              max_new_verbatim: int, max_new_reverse: int, out_path_json: Path) -> list[dict]:
    print(f"\n=== {label} ===  ({len(verbatims)} verbatim + {len(reverses)} reverse)")
    rows: list[dict] = []

    for i, ex in enumerate(verbatims, 1):
        user_msg = ex["messages"][0]["content"]
        gold = ex["messages"][1]["content"]
        lang = ex.get("language", "en")
        pred, sec = generate(tok, model, user_msg, max_new_tokens=max_new_verbatim)
        sim = char_similarity(pred, gold, lang)
        rec = token_recall(pred, gold, lang)
        exact = _normalize(pred, lang) == _normalize(gold, lang)
        rows.append({
            "suite": label, "task": "verbatim", "key": ex.get("article_key"),
            "lang": lang, "prompt": user_msg, "gold": gold, "pred": pred,
            "char_sim": round(sim, 4), "token_recall": round(rec, 4),
            "exact": exact, "len_pred": len(pred), "len_gold": len(gold),
            "gen_seconds": round(sec, 1),
        })
        if i <= 3 or i % 10 == 0 or i == len(verbatims):
            print(f"  [V{i:>3}/{len(verbatims)}] {ex.get('article_key'):>14} {lang}  "
                  f"sim={sim:.2f} rec={rec:.2f} exact={int(exact)}  {sec:.1f}s")
        # save incrementally
        out_path_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    for i, ex in enumerate(reverses, 1):
        user_msg = ex["messages"][0]["content"]
        gold = ex["messages"][1]["content"]
        lang = ex.get("language", "en")
        pred, sec = generate(tok, model, user_msg, max_new_tokens=max_new_reverse)
        expected_n = extract_article_number(gold) or extract_article_number(ex.get("article_key") or "")
        predicted_n = extract_article_number(pred)
        correct = (predicted_n is not None and expected_n is not None
                   and predicted_n == expected_n)
        rows.append({
            "suite": label, "task": "reverse", "key": ex.get("article_key"),
            "lang": lang, "prompt": user_msg, "gold": gold, "pred": pred,
            "expected_n": expected_n, "predicted_n": predicted_n,
            "off_by": (predicted_n - expected_n) if (predicted_n and expected_n) else None,
            "correct": correct, "gen_seconds": round(sec, 1),
        })
        if i <= 3 or i % 10 == 0 or i == len(reverses):
            ok = "✓" if correct else "✗"
            print(f"  [R{i:>3}/{len(reverses)}] gold={ex.get('article_key')} {lang}  "
                  f"pred='{pred[:50]}'  →  {predicted_n} {ok}  {sec:.1f}s")
        out_path_json.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    return rows


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

def _agg_verbatim(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["task"] == "verbatim"]
    if not rows:
        return {}
    n = len(rows)
    en = [r for r in rows if r["lang"] == "en"]
    ar = [r for r in rows if r["lang"] == "ar"]
    def m(rs, k): return sum(r[k] for r in rs) / max(1, len(rs))
    def e(rs):    return sum(1 for r in rs if r["exact"])
    return dict(
        n=n, n_en=len(en), n_ar=len(ar),
        char_sim=m(rows, "char_sim"),
        char_sim_en=m(en, "char_sim") if en else 0,
        char_sim_ar=m(ar, "char_sim") if ar else 0,
        token_recall=m(rows, "token_recall"),
        exact=e(rows), exact_en=e(en), exact_ar=e(ar),
        exact_pct=e(rows) / n,
    )


def _agg_reverse(rows: list[dict]) -> dict:
    rows = [r for r in rows if r["task"] == "reverse"]
    if not rows:
        return {}
    n = len(rows)
    en = [r for r in rows if r["lang"] == "en"]
    ar = [r for r in rows if r["lang"] == "ar"]
    correct = sum(1 for r in rows if r["correct"])
    off_by  = [abs(r["off_by"]) for r in rows if r.get("off_by") is not None]
    within_10 = sum(1 for o in off_by if o <= 10)
    return dict(
        n=n, n_en=len(en), n_ar=len(ar),
        correct=correct, correct_en=sum(1 for r in en if r["correct"]),
        correct_ar=sum(1 for r in ar if r["correct"]),
        accuracy=correct / n,
        within_10=within_10, within_10_pct=within_10 / n,
        median_off_by=(sorted(off_by)[len(off_by) // 2] if off_by else None),
    )


def write_report(rows: list[dict], out_path: Path, args) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Closed-book recall eval — held-out 200-prompt set\n"]
    md.append(f"**Sample:** {args.n_verbatim} verbatim + {args.n_reverse} reverse, "
              f"stratified EN/AR, drawn from `{Path(args.val_jsonl).name}` "
              f"(never seen in training).")
    md.append(f"**Decoding:** greedy (`do_sample=False`), `max_new_tokens` = "
              f"{args.max_new_verbatim} (verbatim), {args.max_new_reverse} (reverse), "
              f"`repetition_penalty=1.0`.")
    md.append(f"**Seed:** {args.seed}.\n")

    suites = sorted({r["suite"] for r in rows})
    md.append("## Aggregate\n")
    md.append("| Suite | Verbatim char-sim | Verbatim exact | Reverse exact | Reverse within ±10 |")
    md.append("|---|---|---|---|---|")
    for s in suites:
        srows = [r for r in rows if r["suite"] == s]
        v = _agg_verbatim(srows); rv = _agg_reverse(srows)
        md.append(
            f"| `{s}` "
            f"| {v.get('char_sim',0):.3f} (EN {v.get('char_sim_en',0):.2f} / AR {v.get('char_sim_ar',0):.2f}) "
            f"| {v.get('exact',0)}/{v.get('n',0)} ({v.get('exact_pct',0)*100:.0f}%) "
            f"| {rv.get('correct',0)}/{rv.get('n',0)} ({rv.get('accuracy',0)*100:.0f}%) "
            f"| {rv.get('within_10',0)}/{rv.get('n',0)} ({rv.get('within_10_pct',0)*100:.0f}%) |"
        )

    for s in suites:
        srows = [r for r in rows if r["suite"] == s]
        md.append(f"\n## Suite: `{s}`\n")
        v = _agg_verbatim(srows); rv = _agg_reverse(srows)

        md.append("### Verbatim")
        md.append(f"- n = {v.get('n',0)}  (EN {v.get('n_en',0)}, AR {v.get('n_ar',0)})")
        md.append(f"- mean char-similarity = **{v.get('char_sim',0):.3f}**  "
                  f"(EN {v.get('char_sim_en',0):.3f}, AR {v.get('char_sim_ar',0):.3f})")
        md.append(f"- mean token-recall = **{v.get('token_recall',0):.3f}**")
        md.append(f"- exact matches = **{v.get('exact',0)} / {v.get('n',0)}** "
                  f"({v.get('exact_pct',0)*100:.1f}%)  "
                  f"(EN {v.get('exact_en',0)}, AR {v.get('exact_ar',0)})")

        md.append("\n### Reverse lookup")
        md.append(f"- n = {rv.get('n',0)}  (EN {rv.get('n_en',0)}, AR {rv.get('n_ar',0)})")
        md.append(f"- exact article-number match = **{rv.get('correct',0)} / {rv.get('n',0)}** "
                  f"({rv.get('accuracy',0)*100:.1f}%)  "
                  f"(EN {rv.get('correct_en',0)}, AR {rv.get('correct_ar',0)})")
        md.append(f"- predictions within ±10 of correct = **{rv.get('within_10',0)} / {rv.get('n',0)}** "
                  f"({rv.get('within_10_pct',0)*100:.1f}%)")
        md.append(f"- median |off-by| = {rv.get('median_off_by','?')}")

        # 5 best + 5 worst verbatim for inspection
        vs = [r for r in srows if r["task"] == "verbatim"]
        if vs:
            md.append("\n#### Best verbatim items (highest char-sim)")
            for r in sorted(vs, key=lambda x: -x["char_sim"])[:5]:
                md.append(f"- **{r['key']}** ({r['lang']}, {r['len_gold']}c) — "
                          f"sim={r['char_sim']:.2f} exact={r['exact']}")
            md.append("\n#### Worst verbatim items (lowest char-sim)")
            for r in sorted(vs, key=lambda x: x["char_sim"])[:5]:
                md.append(f"- **{r['key']}** ({r['lang']}, {r['len_gold']}c) — "
                          f"sim={r['char_sim']:.2f}")

    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport → {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--base-model",  type=str, default=DEFAULT_BASE_MODEL)
    ap.add_argument("--also-base", action="store_true",
                    help="Run the bare base model on the SAME prompts for delta.")
    ap.add_argument("--base-only", action="store_true",
                    help="Skip the adapter; only run the base model.")
    ap.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL,
                    help="Held-out val split to sample prompts from.")
    ap.add_argument("--n-verbatim", type=int, default=100,
                    help="How many kn_verbatim prompts to sample (balanced EN/AR).")
    ap.add_argument("--n-reverse", type=int, default=100,
                    help="How many kn_reverse prompts to sample (balanced EN/AR).")
    ap.add_argument("--max-new-verbatim", type=int, default=600)
    ap.add_argument("--max-new-reverse",  type=int, default=64)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    rows_all_records = [json.loads(l) for l in args.val_jsonl.read_text(encoding="utf-8").splitlines() if l.strip()]
    print(f"Val total: {len(rows_all_records)}  ·  kinds: {Counter(r.get('kind') for r in rows_all_records)}")
    verbatims = stratified_sample(rows_all_records, "kn_verbatim", args.n_verbatim, args.seed)
    reverses  = stratified_sample(rows_all_records, "kn_reverse",  args.n_reverse,  args.seed + 1)
    print(f"Sampled: {len(verbatims)} verbatim ({sum(1 for r in verbatims if r['language']=='en')} EN / "
          f"{sum(1 for r in verbatims if r['language']=='ar')} AR), "
          f"{len(reverses)} reverse ({sum(1 for r in reverses if r['language']=='en')} EN / "
          f"{sum(1 for r in reverses if r['language']=='ar')} AR)")
    json_path = args.out_path.with_suffix(".json")
    args.out_path.parent.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict] = []

    if not args.base_only:
        print(f"\nLoading adapter: {args.adapter_dir}")
        tok, model = load_adapter(args.adapter_dir)
        all_rows.extend(run_suite(tok, model, "adapter", verbatims, reverses,
                                  args.max_new_verbatim, args.max_new_reverse, json_path))
        del model, tok
        torch.cuda.empty_cache()

    if args.base_only or args.also_base:
        print(f"\nLoading base: {args.base_model}")
        tok, model = load_base(args.base_model)
        all_rows.extend(run_suite(tok, model, "base", verbatims, reverses,
                                  args.max_new_verbatim, args.max_new_reverse, json_path))
        del model, tok
        torch.cuda.empty_cache()

    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(all_rows, args.out_path, args)


if __name__ == "__main__":
    main()
