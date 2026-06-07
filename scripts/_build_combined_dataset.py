"""One-off: build qa_pairs_knowledge_scenario.jsonl (train) and
qa_pairs_knowledge_scenario_val.jsonl (val) from:

  - data/qa_pairs_knowledge.jsonl          (22,294 knowledge train records)
  - data/qa_pairs_knowledge_val.jsonl      (928 knowledge val records)
  - data/scenarios_full.jsonl.bak.before_enrich
                                           (4,368 scenario records, 4 per article)

Strategy: keep the existing knowledge train/val as-is, and pull ~5% of the
scenario records into val (so the val loss now also tracks scenario quality
during training) and put the remaining ~95% into train. The val split is
done by sampling whole scenario quartets (4 records per article) so an
article's scenarios never span train and val.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
KN_TRAIN   = ROOT / "data" / "qa_pairs_knowledge.jsonl"
KN_VAL     = ROOT / "data" / "qa_pairs_knowledge_val.jsonl"
SCEN_BAK   = ROOT / "data" / "scenarios_full.jsonl.bak.before_enrich"
OUT_TRAIN  = ROOT / "data" / "qa_pairs_knowledge_scenario.jsonl"
OUT_VAL    = ROOT / "data" / "qa_pairs_knowledge_scenario_val.jsonl"

VAL_FRACTION = 0.05  # roughly match the existing 4.2% knowledge val ratio


def main() -> None:
    kn_train = [json.loads(l) for l in KN_TRAIN.read_text(encoding="utf-8").splitlines() if l.strip()]
    kn_val   = [json.loads(l) for l in KN_VAL.read_text(encoding="utf-8").splitlines() if l.strip()]
    scen     = [json.loads(l) for l in SCEN_BAK.read_text(encoding="utf-8").splitlines() if l.strip()]

    # group scenarios by article
    scen_by_art: dict[str, list[dict]] = defaultdict(list)
    for r in scen:
        scen_by_art[r["article_key"]].append(r)

    arts = sorted(scen_by_art.keys(),
                  key=lambda k: int(k.replace("Article", "").strip()))
    rng = random.Random(20260602)

    # pick whole-article scenario sets for val: VAL_FRACTION of articles
    n_val_arts = max(1, round(len(arts) * VAL_FRACTION))
    val_arts = set(rng.sample(arts, n_val_arts))

    scen_val   = [r for a in val_arts for r in scen_by_art[a]]
    scen_train = [r for r in scen if r["article_key"] not in val_arts]

    out_train = kn_train + scen_train
    out_val   = kn_val   + scen_val

    OUT_TRAIN.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_train) + "\n",
        encoding="utf-8")
    OUT_VAL.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in out_val) + "\n",
        encoding="utf-8")

    print(f"knowledge train: {len(kn_train):>6}  val: {len(kn_val):>5}")
    print(f"scenarios train: {len(scen_train):>6}  val: {len(scen_val):>5}")
    print(f"  (val articles: {len(val_arts)} / {len(arts)} = {100*len(val_arts)/len(arts):.1f}%)")
    print(f"combined  train: {len(out_train):>6} -> {OUT_TRAIN.relative_to(ROOT)}")
    print(f"combined  val:   {len(out_val):>6} -> {OUT_VAL.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
