"""Score the RAFT-RAG pipeline on a question set — cheap, no LLM judge needed.

For each question we run up to three modes and report aggregates:
    closed   no retrieved context (closed-book)            -> the floor
    rag      our hybrid retriever                          -> the product
    oracle   the true article forced into the context      -> the adapter ceiling
This triple separates "is the retriever bad" from "is the adapter bad".

Metrics per (row, mode):
    retrieval_hit     was the expected article in the oracle slot?     (rag only)
    cited_match       does the answer cite the expected article number?
    has_citation      does the answer cite *any* "Article N" / "المادة N"?
    grounding_overlap fraction of answer tokens that appear in the retrieved text
                      (a crude hallucination proxy — low => answer not grounded)
    refusal_ok        for kind=='refusal' rows: does the answer actually refuse?

Optionally logs the aggregates to Weights & Biases (pass a wandb run).

Accepts question rows in either shape:
  * the project's qa_pairs format: {"messages":[{role:user,...},{role:assistant,...}],
    "language": ..., "article_key": "Article N", "kind": "explanation"|"refusal"}
  * a flat shape: {"question": ..., "language": ..., "article_key": ..., "kind": ...}
"""
from __future__ import annotations

import re
from collections import defaultdict

from .index import detect_lang, tokenize
from .infer import RaftRagPipeline

_CITE_RE = re.compile(r"(?:article|art\.?|المادة|مادة)\s*[#:]?\s*0*([0-9٠-٩]{1,4})", re.IGNORECASE)
_AR2EN_DIGITS = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_REFUSAL_CUES = (
    "cannot provide legal advice", "can't provide legal advice", "not a lawyer",
    "consult a qualified", "qualified attorney", "qualified lawyer",
    "لا أستطيع تقديم", "لا يمكنني تقديم", "استشارة قانونية", "محامٍ مؤهل", "محام مؤهل",
)


def _row_fields(row: dict) -> tuple[str, str, str | None, str]:
    if "messages" in row:
        q = row["messages"][0]["content"]
        kind = row.get("kind", "explanation")
        return q, row.get("language") or detect_lang(q), row.get("article_key"), kind
    q = row.get("question") or row.get("prompt") or ""
    return q, row.get("language") or detect_lang(q), row.get("article_key"), row.get("kind", "explanation")


def _expected_num(article_key: str | None) -> str | None:
    if not article_key:
        return None
    m = re.fullmatch(r"\s*Article\s*(\d+)\s*", article_key)
    return str(int(m.group(1))) if m else None


def _cited_numbers(text: str) -> set[str]:
    out = set()
    for raw in _CITE_RE.findall(text or ""):
        d = raw.translate(_AR2EN_DIGITS)
        if d.isdigit():
            out.add(str(int(d)))
    return out


def _grounding_overlap(answer: str, context_text: str) -> float:
    a = set(tokenize(answer))
    if not a:
        return 0.0
    return round(len(a & set(tokenize(context_text))) / len(a), 4)


def _is_refusal(text: str) -> bool:
    low = (text or "").lower()
    return any(cue.lower() in low for cue in _REFUSAL_CUES)


def evaluate(cfg: dict, rows: list[dict], *, modes=("closed", "rag", "oracle"),
             limit: int | None = None, wandb_run=None) -> dict:
    if limit:
        rows = rows[:limit]
    pipe = RaftRagPipeline.from_config(cfg, load_model=True)
    idx = pipe.retriever.index

    per_row = []
    agg: dict[str, dict[str, list]] = {m: defaultdict(list) for m in modes}

    for i, row in enumerate(rows, 1):
        q, lang, art_key, kind = _row_fields(row)
        exp_num = _expected_num(art_key)
        rec = {"i": i, "lang": lang, "kind": kind, "article_key": art_key,
               "question": q[:200], "by_mode": {}}
        for m in modes:
            if m == "oracle" and not art_key:
                continue
            r = pipe.ask(q, lang=lang, mode=m, force_key=(art_key if m == "oracle" else None))
            ans = r.get("answer") or ""
            ctx_keys = r.get("retrieved_oracle", [])
            ctx_text = " ".join((idx.get(k).text(lang) if idx.get(k) else "") for k in ctx_keys)
            cited = _cited_numbers(ans)
            retrieved_nums = {_expected_num(k) for k in ctx_keys} - {None}
            scored = {
                "retrieved_oracle": ctx_keys,
                "retrieval_hit": (exp_num in retrieved_nums) if (m == "rag" and exp_num) else None,
                "cited_match": (exp_num in cited) if exp_num else None,
                "has_citation": bool(cited),
                "grounding_overlap": (_grounding_overlap(ans, ctx_text) if ctx_text else None),
                "refusal_ok": (_is_refusal(ans) if kind == "refusal" else None),
                "answer": ans[:600],
                "gen_seconds": r.get("gen_seconds"),
            }
            rec["by_mode"][m] = scored
            for key, val in scored.items():
                if isinstance(val, bool):
                    agg[m][key].append(1.0 if val else 0.0)
                elif key == "grounding_overlap" and isinstance(val, (int, float)):
                    agg[m][key].append(float(val))
        per_row.append(rec)
        cite_bits = " ".join(f"{m}:cite={rec['by_mode'].get(m, {}).get('cited_match')}"
                             for m in modes if m in rec["by_mode"])
        print(f"[{i}/{len(rows)}] {kind} {art_key} | {cite_bits}")

    summary = {m: {k: round(sum(v) / len(v), 4) for k, v in d.items() if v} for m, d in agg.items()}
    out = {"summary": summary, "n_rows": len(rows), "modes": list(modes), "rows": per_row}

    if wandb_run is not None:
        flat = {f"{m}/{k}": v for m, d in summary.items() for k, v in d.items()}
        for k, v in flat.items():
            wandb_run.summary[k] = v
        wandb_run.log(flat)
    return out
