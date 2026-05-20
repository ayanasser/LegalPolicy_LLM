"""Build the COMBINED dataset = knowledge + house-style, for the "both skills" run.

Concatenates the two existing datasets without modification:
  - data/qa_pairs_knowledge.jsonl (+ _val.jsonl) — Stage A knowledge-injection,
    21,679 + 903 examples, 10 task families, 100 % article coverage, ~20× per article.
    Trains the model on WHAT the law says (verbatim recall, completion, fill-the-gap,
    reverse lookup, placement, cross-language translate, bilingual, reference card,
    contrast, roster). NO house-style template in these.
  - data/qa_pairs.jsonl (+ _val.jsonl) — Stage 1 house-style SFT, 3,389 + 592 examples.
    Trains the model on HOW to explain (1-line summary → "Article X provides:" → ≥3
    bullets → worked example → DISCLAIMER), and includes refusal seeds.

The combined dataset is shuffled (deterministic seed), de-duplicated on the
(prompt-prefix, response-prefix) signature, and written to
data/qa_pairs_combined.jsonl + qa_pairs_combined_val.jsonl in the same ChatML
schema that train_unsloth.py / train.py consume.

Why combine instead of training sequentially (knowledge → then house-style)?
A sequential second pass on qa_pairs.jsonl would optimise paraphrased outputs that
contradict the verbatim outputs the model just learned, risking catastrophic
forgetting of the closed-book recall (the thesis result). Mixing both signals in
one pass keeps the optimiser honest to both distributions simultaneously, with
no forgetting because there's no "later stage" where one distribution is absent.

The 7:1 ratio (≈21.7k knowledge : 3.4k style) is intentional: knowledge dominates
as the thesis-critical signal, but style examples are mechanically distinct
enough (very different prompt shapes and ~250 w gold answers vs ~30 w short text)
that they aren't drowned out — the model learns "when the prompt asks 'explain',
do house style; when it asks 'quote' / 'complete' / 'which article', do verbatim".

Usage:
    python scripts/build_dataset_combined.py \\
        --config src/legal_explainer/finetune/configs/qlora_qwen3b_combined.yaml
    # smoke test (caps both sources):
    python scripts/build_dataset_combined.py --config <cfg> --max-each 200
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_KNOWLEDGE_TRAIN = PROJECT_ROOT / "data" / "qa_pairs_knowledge.jsonl"
DEFAULT_KNOWLEDGE_VAL   = PROJECT_ROOT / "data" / "qa_pairs_knowledge_val.jsonl"
DEFAULT_STYLE_TRAIN     = PROJECT_ROOT / "data" / "qa_pairs.jsonl"
DEFAULT_STYLE_VAL       = PROJECT_ROOT / "data" / "qa_pairs_val.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        raise FileNotFoundError(f"Source file not found: {path}")
    rows = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return rows


def _tag(rows: list[dict], source: str) -> list[dict]:
    """Attach a top-level `source` field so we can audit the mix later."""
    for r in rows:
        r.setdefault("source", source)
    return rows


def _dedup(rows: list[dict]) -> list[dict]:
    """Dedup on (instruction-prefix, response-prefix) — same approach as the other builders."""
    seen, unique = set(), []
    for r in rows:
        msgs = r.get("messages") or []
        if len(msgs) < 2:
            continue
        sig = (msgs[0]["content"][:200], msgs[1]["content"][:200])
        if sig in seen:
            continue
        seen.add(sig)
        unique.append(r)
    return unique


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _summary(label: str, rows: list[dict]) -> None:
    kinds = Counter(r.get("kind") for r in rows)
    langs = Counter(r.get("language") for r in rows)
    sources = Counter(r.get("source") for r in rows)
    print(f"  {label}: {len(rows)} examples")
    print(f"    by source: {dict(sources)}")
    print(f"    by kind:   {dict(sorted(kinds.items()))}")
    print(f"    by lang:   {dict(sorted(langs.items()))}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path,
                    help="YAML config; only its data.train_jsonl / data.val_jsonl are used here.")
    ap.add_argument("--knowledge-train", type=Path, default=DEFAULT_KNOWLEDGE_TRAIN)
    ap.add_argument("--knowledge-val",   type=Path, default=DEFAULT_KNOWLEDGE_VAL)
    ap.add_argument("--style-train",     type=Path, default=DEFAULT_STYLE_TRAIN)
    ap.add_argument("--style-val",       type=Path, default=DEFAULT_STYLE_VAL)
    ap.add_argument("--max-each", type=int, default=0,
                    help="If > 0, cap each source at this many rows (smoke test).")
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    train_out = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_out   = PROJECT_ROOT / cfg["data"]["val_jsonl"]

    print("Loading sources ...")
    kn_train = _tag(_read_jsonl(args.knowledge_train), source="knowledge")
    kn_val   = _tag(_read_jsonl(args.knowledge_val),   source="knowledge")
    st_train = _tag(_read_jsonl(args.style_train),     source="style")
    st_val   = _tag(_read_jsonl(args.style_val),       source="style")
    _summary("knowledge/train", kn_train)
    _summary("knowledge/val",   kn_val)
    _summary("style/train",     st_train)
    _summary("style/val",       st_val)

    if args.max_each:
        rng_cap = random.Random(args.seed)
        for rows in (kn_train, kn_val, st_train, st_val):
            rng_cap.shuffle(rows)
        kn_train, kn_val = kn_train[:args.max_each], kn_val[:max(1, args.max_each // 20)]
        st_train, st_val = st_train[:args.max_each], st_val[:max(1, args.max_each // 20)]
        print(f"  [smoke] capped each source at {args.max_each}")

    train = _dedup(kn_train + st_train)
    val   = _dedup(kn_val + st_val)

    rng = random.Random(args.seed)
    rng.shuffle(train); rng.shuffle(val)

    print("\nCombined splits:")
    _summary("train", train)
    _summary("val",   val)

    _write_jsonl(train, train_out)
    _write_jsonl(val,   val_out)
    print(f"\nTrain: {len(train):>6} -> {train_out}")
    print(f"Val:   {len(val):>6} -> {val_out}")


if __name__ == "__main__":
    main()
