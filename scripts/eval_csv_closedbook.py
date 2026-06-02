"""Closed-book Q/A eval against a CSV of (question, answer, legal_reference) rows.

Loads a PEFT/QLoRA adapter (4-bit base), generates an answer for each question
with hardened decoding params (low temp, repetition penalty, no-repeat-ngram —
the fix for the bullet-loops / degeneration seen earlier), and scores
"relatedness" against the gold answer with cheap heuristics — no LLM judge.

Per-row scores
  cited_article_match     model output cites "مادة N" matching legal_reference
  has_citation            model output cites *any* مادة / Article
  token_overlap_gold      Jaccard over Arabic/Latin word tokens vs gold answer
                          (proxy for "is the output related to the gold answer")
  refused                 model output looks like a refusal (per a small cue list)
  empty                   model output is empty / near-empty

Output: a JSON with the per-row records + an aggregate summary, and a small
console table.

    python scripts/eval_csv_closedbook.py \
        --adapter runs/qlora-qwen2.5-3b-combined \
        --csv data/general_user_legal_questions.csv \
        --out reports/eval/general_user_legal_questions.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from peft import AutoPeftModelForCausalLM
from transformers import AutoTokenizer, BitsAndBytesConfig

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Arabic-aware tokeniser and digit normaliser
_WORD_RE = re.compile(r"\w+", re.UNICODE)
_AR2EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
# "Article 5", "Art. 5", "المادة ٥", "مادة 5"
_CITE_RE = re.compile(r"(?:article|art\.?|المادة|مادة)\s*[#:]?\s*0*([0-9٠-٩]{1,4})", re.IGNORECASE)
# common Arabic stop-words to keep Jaccard from being noise-dominated
_STOP = {
    # Arabic
    "في", "من", "إلى", "على", "عن", "أن", "إن", "أو", "و", "ال", "هذا", "هذه",
    "ذلك", "تلك", "هو", "هي", "ما", "ماذا", "كيف", "متى", "أين", "لماذا",
    "كان", "كانت", "يكون", "تكون", "قد", "لو", "لا", "نعم", "بس", "ده", "دي",
    "كل", "مع", "بين", "تحت", "فوق", "بعد", "قبل", "حيث", "إذا", "أم",
    # Latin
    "the", "a", "an", "of", "to", "in", "and", "or", "is", "are", "be", "was",
    "were", "for", "on", "with", "as", "by", "this", "that", "it", "if", "from",
}


def tokens(text: str) -> set[str]:
    return {t for t in _WORD_RE.findall((text or "").lower()) if t not in _STOP and len(t) > 1}


def cited_nums(text: str) -> set[str]:
    out: set[str] = set()
    for raw in _CITE_RE.findall(text or ""):
        d = raw.translate(_AR2EN_DIGITS)
        if d.isdigit():
            out.add(str(int(d)))
    return out


def parse_ref_num(ref: str) -> str | None:
    if not ref:
        return None
    m = re.search(r"([0-9٠-٩]{1,4})", str(ref))
    if not m:
        return None
    d = m.group(1).translate(_AR2EN_DIGITS)
    return str(int(d)) if d.isdigit() else None


_REFUSAL_CUES = (
    "لا أستطيع", "لا يمكنني", "لست محام", "لست محامي", "لست متخصص",
    "استشارة قانونية", "استشر محام", "محامٍ مؤهل", "محامي مؤهل",
    "i am not a lawyer", "i'm not a lawyer", "cannot provide legal",
    "consult a qualified", "qualified attorney",
)


def looks_like_refusal(text: str) -> bool:
    t = (text or "").lower()
    return any(cue.lower() in t for cue in _REFUSAL_CUES)


_AR_CHARS = re.compile(r"[؀-ۿ]")


def detect_lang(text: str) -> str:
    """'ar' if predominantly Arabic script, else 'en'. Matches the heuristic
    used at training time by scripts/prefix_language_tag.py."""
    if not text:
        return "en"
    non_space = len(text.replace(" ", "")) or 1
    return "ar" if len(_AR_CHARS.findall(text)) >= 0.20 * non_space else "en"


def language_tag(question: str) -> str:
    """Match the prompt shape the adapter was trained on: a leading [AR] / [EN]
    tag prepended to the user message (see scripts/prefix_language_tag.py)."""
    lang = detect_lang(question)
    return "[AR] " if lang == "ar" else "[EN] "


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    return round(len(a & b) / max(1, len(a | b)), 4)


def load_model(adapter_dir: Path):
    print(f"Loading adapter: {adapter_dir}")
    tok = AutoTokenizer.from_pretrained(str(adapter_dir), trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                             bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoPeftModelForCausalLM.from_pretrained(
        str(adapter_dir), quantization_config=bnb, device_map="auto", trust_remote_code=True)
    model.eval()
    return model, tok


def generate(model, tok, question: str, *, max_new_tokens: int, temperature: float,
             top_p: float, repetition_penalty: float, no_repeat_ngram_size: int) -> tuple[str, float]:
    chat = [{"role": "user", "content": question}]
    prompt = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)
    plen = inputs["input_ids"].shape[1]
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=max_new_tokens,
            do_sample=temperature > 0, temperature=max(temperature, 1e-5), top_p=top_p,
            repetition_penalty=repetition_penalty, no_repeat_ngram_size=no_repeat_ngram_size,
            pad_token_id=(tok.pad_token_id or tok.eos_token_id),
        )
    dt = time.perf_counter() - t0
    return tok.decode(out[0, plen:], skip_special_tokens=True).strip(), round(dt, 2)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", required=True, type=Path)
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--top-p", type=float, default=0.85)
    ap.add_argument("--repetition-penalty", type=float, default=1.15)
    ap.add_argument("--no-repeat-ngram-size", type=int, default=3)
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--language-tag", choices=["auto", "off"], default="auto",
                    help="Prepend [AR]/[EN] to each question to match training. "
                         "Adapters trained on a language-tagged dataset (e.g. "
                         "qa_pairs_knowledge_words.jsonl after prefix_language_tag.py) "
                         "expect this; older un-tagged adapters do not.")
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    if args.limit:
        rows = rows[:args.limit]
    print(f"CSV: {len(rows)} rows")

    model, tok = load_model(args.adapter)
    print("Generating ...\n")

    out_rows = []
    sums = {"cited_article_match": 0, "has_citation": 0, "refused": 0, "empty": 0,
            "token_overlap_gold": 0.0}
    for i, row in enumerate(rows, 1):
        q = (row.get("question") or "").strip()
        gold = (row.get("answer") or "").strip()
        ref_num = parse_ref_num(row.get("legal_reference") or "")
        q_for_model = (language_tag(q) + q) if args.language_tag == "auto" else q
        pred, dt = generate(
            model, tok, q_for_model,
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
        )
        cited = cited_nums(pred)
        cited_match = bool(ref_num) and ref_num in cited
        has_citation = bool(cited)
        refused = looks_like_refusal(pred)
        empty = len(pred) < 5
        overlap = jaccard(tokens(pred), tokens(gold))

        rec = {
            "id": row.get("id"), "question": q, "gold_answer": gold,
            "legal_reference": row.get("legal_reference"), "expected_article": ref_num,
            "prediction": pred, "gen_seconds": dt,
            "cited_articles": sorted(cited), "cited_article_match": cited_match,
            "has_citation": has_citation, "refused": refused, "empty": empty,
            "token_overlap_gold": overlap,
        }
        out_rows.append(rec)

        for k in ("cited_article_match", "has_citation", "refused", "empty"):
            sums[k] += int(rec[k])
        sums["token_overlap_gold"] += overlap

        print(f"[{i:>2}/{len(rows)}] id={row.get('id')} ref={ref_num}  "
              f"cite={cited_match!s:5s}  refused={refused!s:5s}  "
              f"overlap={overlap:.2f}  {dt:.1f}s")

    n = max(1, len(out_rows))
    agg = {
        "n_rows": len(out_rows),
        "cited_article_match_rate": round(sums["cited_article_match"] / n, 4),
        "has_citation_rate": round(sums["has_citation"] / n, 4),
        "refusal_rate": round(sums["refused"] / n, 4),
        "empty_rate": round(sums["empty"] / n, 4),
        "mean_token_overlap_gold": round(sums["token_overlap_gold"] / n, 4),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({
        "adapter": str(args.adapter), "csv": str(args.csv),
        "gen_params": {"max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
                       "top_p": args.top_p, "repetition_penalty": args.repetition_penalty,
                       "no_repeat_ngram_size": args.no_repeat_ngram_size},
        "summary": agg, "rows": out_rows,
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== summary ===")
    for k, v in agg.items():
        print(f"  {k:30s} {v}")
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
