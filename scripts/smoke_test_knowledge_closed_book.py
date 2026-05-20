"""Quick closed-book smoke test for the Stage-A knowledge adapter.

Loads runs/qlora-qwen2.5-1.5b-knowledge/ (or any QLoRA adapter directory) and
asks it to:
  1. Quote ~8 selected articles VERBATIM (4 EN + 4 AR), with NO article text in
     the prompt — pure memorisation test.
  2. Identify ~3 articles given only a verbatim snippet — reverse-lookup test.

For each item we compute a cheap text-similarity score against the ground-truth
article text from data/orig_data.json (after the same light cleanup used during
dataset building), and for reverse lookups whether the model emitted the correct
article number.

Greedy decoding (do_sample=False, repetition_penalty=1.0) so the numbers reflect
what the weights actually know, not lucky sampling. ~30-40s per generation on a
6 GB RTX 3050, ~6-8 min wall-clock for the default 11-item suite.

Usage:
    python scripts/smoke_test_knowledge_closed_book.py
    python scripts/smoke_test_knowledge_closed_book.py \
        --adapter-dir runs/qlora-qwen2.5-1.5b-knowledge \
        --out-path reports/eval/knowledge_smoke.md

Add --base-only to evaluate the bare Qwen 2.5 1.5B Instruct base model with no
adapter (same prompts, same decoding) for a base-vs-finetuned delta.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from difflib import SequenceMatcher
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from legal_explainer.finetune.knowledge_builder import clean_text  # noqa: E402

DEFAULT_ADAPTER_DIR = PROJECT_ROOT / "runs" / "qlora-qwen2.5-1.5b-knowledge"
DEFAULT_BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
DEFAULT_CORPUS = PROJECT_ROOT / "data" / "orig_data.json"
DEFAULT_OUT = PROJECT_ROOT / "reports" / "eval" / "knowledge_smoke.md"

# Test articles chosen to span the corpus and lengths.
VERBATIM_TESTS: list[tuple[str, str]] = [
    # (article_key, language)
    ("Article 1", "en"),        # foundational, multi-paragraph
    ("Article 280", "en"),      # mid, solidarity between creditors
    ("Article 775", "en"),      # short, suretyship
    ("Article 1068", "en"),     # late, mortgage purge procedure
    ("Article 17", "ar"),       # inheritance conflict-of-laws
    ("Article 836", "ar"),      # partition
    ("Article 990", "ar"),      # usufruct
    ("Article 1112", "ar"),     # pledge
]

# (article_key, language, snippet_chars)  — the snippet is drawn from the cleaned
# article text and inserted into the reverse-lookup prompt; gold answer is the key.
REVERSE_TESTS: list[tuple[str, str, int]] = [
    ("Article 775", "en", 90),
    ("Article 1", "en", 160),
    ("Article 775", "ar", 60),
]


# ---------------------------------------------------------------------------
# scoring helpers
# ---------------------------------------------------------------------------

_WS = re.compile(r"\s+")
_PUNCT = re.compile(r"[،؛؟.,;:!?\"'()\[\]{}«»\-—–]")


def _normalize_for_match(text: str, lang: str) -> str:
    t = clean_text(text)
    t = _PUNCT.sub(" ", t)
    t = _WS.sub(" ", t).strip()
    if lang == "en":
        t = t.lower()
    return t


def char_similarity(pred: str, gold: str, lang: str) -> float:
    p = _normalize_for_match(pred, lang)
    g = _normalize_for_match(gold, lang)
    if not p or not g:
        return 0.0
    return SequenceMatcher(None, p, g).ratio()


def gold_token_recall(pred: str, gold: str, lang: str) -> float:
    """Fraction of distinct gold tokens (len>=4) appearing in the prediction."""
    p = set(_normalize_for_match(pred, lang).split())
    g_tokens = [w for w in _normalize_for_match(gold, lang).split() if len(w) >= 4]
    if not g_tokens:
        return 0.0
    g_unique = set(g_tokens)
    return len(g_unique & p) / len(g_unique)


_ARTICLE_NUM_RE = re.compile(r"(?:Article|article|المادة|المادّة)\s*[#:]?\s*([0-9]+)")


def extract_article_number(text: str) -> int | None:
    m = _ARTICLE_NUM_RE.search(text or "")
    return int(m.group(1)) if m else None


# ---------------------------------------------------------------------------
# prompt builders — same wording families as the training set
# ---------------------------------------------------------------------------

def verbatim_prompt(n: str, lang: str) -> str:
    if lang == "en":
        return f"Quote Article {n} of the Egyptian Civil Code exactly as it is written."
    return f"اذكر نص المادة {n} من القانون المدني المصري حرفياً."


def reverse_prompt(snippet: str, lang: str) -> str:
    if lang == "en":
        return (
            "Which article of the Egyptian Civil Code contains the following provision? "
            "Reply with the article number only.\n\n"
            f'"{snippet}"'
        )
    return (
        "أي مادة من القانون المدني المصري تتضمن النص التالي؟ اذكر رقم المادة فقط.\n\n"
        f'"{snippet}"'
    )


# ---------------------------------------------------------------------------
# model loading / generation
# ---------------------------------------------------------------------------

def load_adapter(adapter_dir: Path):
    from peft import AutoPeftModelForCausalLM
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir), quantization_config=bnb,
        device_map="auto", trust_remote_code=True,
    )
    model.eval()
    return tok, model


def load_base(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(
        load_in_4bit=True, bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=bnb,
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
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,             # greedy — measure what the weights know
            repetition_penalty=1.0,
            pad_token_id=tok.eos_token_id,
        )
    elapsed = time.perf_counter() - t0
    pred = tok.decode(out[0, prompt_len:], skip_special_tokens=True).strip()
    return pred, elapsed


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def run_suite(tok, model, corpus, label: str, max_new_tokens: int) -> list[dict]:
    print(f"\n=== {label} ===")
    rows: list[dict] = []
    for i, (key, lang) in enumerate(VERBATIM_TESTS, 1):
        entry = corpus[key]
        gold = clean_text(entry["english" if lang == "en" else "arabic"])
        prompt = verbatim_prompt(key.split()[1], lang)
        pred, sec = generate(tok, model, prompt, max_new_tokens=max_new_tokens)
        sim = char_similarity(pred, gold, lang)
        rec = gold_token_recall(pred, gold, lang)
        rows.append({
            "suite": label, "task": "verbatim", "key": key, "lang": lang,
            "prompt": prompt, "gold": gold, "pred": pred,
            "char_sim": round(sim, 3), "token_recall": round(rec, 3),
            "exact": _normalize_for_match(pred, lang) == _normalize_for_match(gold, lang),
            "len_pred": len(pred), "len_gold": len(gold), "gen_seconds": round(sec, 1),
        })
        print(f"  [V{i}] {key:>12} {lang}  sim={sim:.2f}  recall={rec:.2f}  "
              f"pred={len(pred)}c gold={len(gold)}c  {sec:.1f}s")

    for i, (key, lang, snip_chars) in enumerate(REVERSE_TESTS, 1):
        entry = corpus[key]
        gold_text = clean_text(entry["english" if lang == "en" else "arabic"])
        snippet = gold_text[:snip_chars]
        prompt = reverse_prompt(snippet, lang)
        pred, sec = generate(tok, model, prompt, max_new_tokens=64)
        expected_n = int(key.split()[1])
        predicted_n = extract_article_number(pred)
        correct = predicted_n == expected_n
        rows.append({
            "suite": label, "task": "reverse", "key": key, "lang": lang,
            "prompt": prompt, "gold": f"Article {expected_n}", "pred": pred,
            "predicted_n": predicted_n, "expected_n": expected_n, "correct": correct,
            "gen_seconds": round(sec, 1),
        })
        print(f"  [R{i}] gold={key} lang={lang}  pred='{pred[:80]}'  "
              f"-> {predicted_n}  {'✓' if correct else '✗'}  {sec:.1f}s")
    return rows


def write_report(rows: list[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md = ["# Closed-book smoke test — knowledge adapter\n"]
    suites = {r["suite"] for r in rows}
    for s in sorted(suites):
        srows = [r for r in rows if r["suite"] == s]
        verbatims = [r for r in srows if r["task"] == "verbatim"]
        reverses = [r for r in srows if r["task"] == "reverse"]
        md.append(f"\n## Suite: `{s}`\n")
        md.append("**Aggregate**:")
        if verbatims:
            avg_sim = sum(r["char_sim"] for r in verbatims) / len(verbatims)
            avg_rec = sum(r["token_recall"] for r in verbatims) / len(verbatims)
            n_exact = sum(1 for r in verbatims if r["exact"])
            md.append(f"- verbatim ({len(verbatims)} items): "
                      f"mean char-sim **{avg_sim:.2f}**, "
                      f"mean token-recall **{avg_rec:.2f}**, "
                      f"exact matches **{n_exact}/{len(verbatims)}**")
        if reverses:
            n_ok = sum(1 for r in reverses if r["correct"])
            md.append(f"- reverse  ({len(reverses)} items): correct article-num "
                      f"**{n_ok}/{len(reverses)}**")
        md.append("")
        md.append("### Verbatim recall (closed-book)")
        for r in verbatims:
            md.append(f"\n#### {r['key']} ({r['lang']}) — char-sim {r['char_sim']:.2f}, "
                      f"token-recall {r['token_recall']:.2f}, exact={r['exact']}")
            md.append(f"**Prompt:** {r['prompt']}")
            md.append(f"\n**Gold ({r['len_gold']}c):**\n```\n{r['gold']}\n```")
            md.append(f"\n**Prediction ({r['len_pred']}c, {r['gen_seconds']}s):**\n```\n{r['pred']}\n```")
        md.append("\n### Reverse lookup (closed-book)")
        for r in reverses:
            ok = "✓" if r["correct"] else "✗"
            md.append(f"\n#### {ok} expected={r['expected_n']} predicted={r['predicted_n']} "
                      f"({r['lang']}, {r['gen_seconds']}s)")
            md.append(f"**Prompt:** {r['prompt']}")
            md.append(f"\n**Prediction:** `{r['pred']}`")
    out_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\nReport written -> {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter-dir", type=Path, default=DEFAULT_ADAPTER_DIR)
    ap.add_argument("--base-only", action="store_true",
                    help="Skip the adapter; evaluate the bare base model.")
    ap.add_argument("--also-base", action="store_true",
                    help="Run base too (sequentially, after the adapter) for comparison.")
    ap.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    ap.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    ap.add_argument("--out-path", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    corpus = json.loads(args.corpus.read_text(encoding="utf-8"))
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM={torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    all_rows: list[dict] = []
    if not args.base_only:
        print(f"Loading adapter: {args.adapter_dir}")
        tok, model = load_adapter(args.adapter_dir)
        all_rows.extend(run_suite(tok, model, corpus, "knowledge_adapter", args.max_new_tokens))
        del model, tok
        torch.cuda.empty_cache()

    if args.base_only or args.also_base:
        print(f"Loading base model: {args.base_model}")
        tok, model = load_base(args.base_model)
        all_rows.extend(run_suite(tok, model, corpus, "base_no_adapter", args.max_new_tokens))
        del model, tok
        torch.cuda.empty_cache()

    # also dump raw json next to the markdown report
    json_path = args.out_path.with_suffix(".json")
    json_path.write_text(json.dumps(all_rows, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(all_rows, args.out_path)
    print(f"Raw json -> {json_path}")


if __name__ == "__main__":
    main()
