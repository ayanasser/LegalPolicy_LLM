"""Swap qa_pairs.jsonl's small refusal block for the augmented refusal set.

Loads `data/qa_pairs.jsonl` (and `..._val.jsonl`), drops the existing `kind=refusal`
records, then builds new refusal records from `refusal_seeds_v2.yaml` (40 EN + 40 AR
seeds, each expanded by 8 paraphrase wrappers → 640 records, shuffled + split 85/15
into train/val so the val set also gets a refusal sample).

Deterministic: no API calls. Preserves all `kind=explanation` records in their
original order; only refusal records are replaced and the file is re-shuffled.

Usage:
    python scripts/inject_refusals_into_qa_pairs.py
    # or override the seed file / variants:
    python scripts/inject_refusals_into_qa_pairs.py --refusal-variants 8 \\
        --refusal-seeds refusal_seeds_v2.yaml
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.knowledge_builder import load_refusal_records  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-jsonl", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs.jsonl")
    ap.add_argument("--val-jsonl", type=Path,
                    default=PROJECT_ROOT / "data" / "qa_pairs_val.jsonl")
    ap.add_argument("--refusal-seeds", type=str, default="refusal_seeds_v2.yaml")
    ap.add_argument("--refusal-variants", type=int, default=8)
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=13)
    args = ap.parse_args()

    train = _read_jsonl(args.train_jsonl)
    val = _read_jsonl(args.val_jsonl)
    print(f"Loaded: train={len(train)}  val={len(val)}")

    def split(rows): return [r for r in rows if r.get("kind") != "refusal"], \
                            [r for r in rows if r.get("kind") == "refusal"]
    train_other, train_refusals = split(train)
    val_other,   val_refusals   = split(val)
    print(f"  → non-refusal: train={len(train_other)}  val={len(val_other)}")
    print(f"  → refusals dropped: train={len(train_refusals)}  val={len(val_refusals)}")

    new_refusals = load_refusal_records(args.refusal_seeds, args.refusal_variants, args.seed)
    print(f"Built {len(new_refusals)} new refusal records "
          f"({args.refusal_variants}x augmentation from {args.refusal_seeds}).")

    # 85/15 split on the new refusals so val also gets a refusal sample
    rng = random.Random(args.seed)
    rng.shuffle(new_refusals)
    n_val = max(1, int(len(new_refusals) * args.val_fraction))
    new_val_refusals = new_refusals[:n_val]
    new_train_refusals = new_refusals[n_val:]

    train_combined = train_other + new_train_refusals
    val_combined = val_other + new_val_refusals
    rng.shuffle(train_combined)
    rng.shuffle(val_combined)

    print(f"\nNew totals:")
    print(f"  train={len(train_combined)}  ({Counter(r.get('kind') for r in train_combined)})")
    print(f"  val  ={len(val_combined)}  ({Counter(r.get('kind') for r in val_combined)})")

    _write_jsonl(train_combined, args.train_jsonl)
    _write_jsonl(val_combined,   args.val_jsonl)
    print(f"\nWrote → {args.train_jsonl}")
    print(f"Wrote → {args.val_jsonl}")


if __name__ == "__main__":
    main()
