"""One-off: append ONE knowledge-task record per article to scenarios_full.jsonl.

For each of the 1,092 articles already in scenarios_full.jsonl, pick exactly
one record from qa_pairs_knowledge.jsonl, rotating across the 10 task families
(kn_verbatim, kn_complete, kn_gap, kn_reverse, kn_placement, kn_translate,
kn_bilingual, kn_card, kn_contrast, kn_roster) in round-robin order by article
number so each family gets ~109 articles. If the target family has no record
for that article, fall back through the remaining families in order. The
appended records use the EXACT same schema as the existing scenarios
(messages / language / kind / article_key) so no downstream code changes.
"""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCEN = ROOT / "data" / "scenarios_full.jsonl"
KN   = ROOT / "data" / "qa_pairs_knowledge.jsonl"

FAMILIES = [
    "kn_verbatim", "kn_complete", "kn_gap", "kn_reverse", "kn_placement",
    "kn_translate", "kn_bilingual", "kn_card", "kn_contrast",
    # kn_roster excluded as a primary target: its records are topic-based
    # (article_key is null), so they can't be matched per-article.
]

rng = random.Random(20260602)


def article_num(key: str) -> int:
    s = key.replace("Article", "").strip()
    return int(s) if s.isdigit() else 10 ** 9


def main() -> None:
    scen_rows = [json.loads(l) for l in SCEN.read_text(encoding="utf-8").splitlines() if l.strip()]
    kn_rows = [json.loads(l) for l in KN.read_text(encoding="utf-8").splitlines() if l.strip()]

    # Index knowledge records by (article_key, kind)
    by_art_kind: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in kn_rows:
        by_art_kind[(r["article_key"], r["kind"])].append(r)

    # Articles present in scenarios, sorted by article number for stable round-robin
    scen_articles = sorted({r["article_key"] for r in scen_rows}, key=article_num)

    print(f"scenarios articles: {len(scen_articles)}")
    print(f"families:           {len(FAMILIES)}  ({len(scen_articles) / len(FAMILIES):.1f} articles each)")

    appended = []
    family_used: dict[str, int] = defaultdict(int)
    fallbacks = 0

    for i, art in enumerate(scen_articles):
        primary = FAMILIES[i % len(FAMILIES)]
        # Try primary first, then fall back through a SHUFFLED list of the
        # remaining families so fallbacks don't all pile onto one family.
        rest = [f for f in FAMILIES if f != primary]
        rng.shuffle(rest)
        order = [primary] + rest
        picked = None
        for fam in order:
            pool = by_art_kind.get((art, fam))
            if pool:
                picked = rng.choice(pool)
                family_used[fam] += 1
                if fam != primary:
                    fallbacks += 1
                break
        if picked is None:
            print(f"  WARN: no knowledge record for {art} in any family")
            continue
        appended.append(picked)

    # Append, preserving order (scenarios first, then enrichments — same shape).
    with SCEN.open("a", encoding="utf-8") as f:
        for r in appended:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nappended: {len(appended)} records")
    print(f"fallbacks needed: {fallbacks} articles")
    print("per-family counts:")
    for fam in FAMILIES:
        print(f"  {fam:14s}  {family_used[fam]}")


if __name__ == "__main__":
    main()
