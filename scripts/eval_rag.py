"""RAGAS-style evaluation for the Graph RAG, Bilingual RAG and fine-tuned systems.

Metrics: faithfulness, answer relevance, context precision, context recall,
answer correctness (LLM-judged, 0–1) + citation accuracy, retrieval hit@k, MRR
(deterministic). Judged by a local Ollama model (default) or by Claude Code.

Simple, flat output under reports/eval/rag/:
  <system>__<dataset>.jsonl   one run = one file (answers + every metric per row)
  SUMMARY.md                  one table, one row per run  ← read this
  SUMMARY.json                machine-readable summary

Examples
--------
  # Bilingual RAG over all three gold sets, judged locally, end-to-end:
  python scripts/eval_rag.py --system bilingual-rag

  # Just one dataset, first 20 rows:
  python scripts/eval_rag.py --system bilingual-rag \
      --dataset data/article_lookup_golden.csv --limit 20

  # Judge with Claude Code instead of Ollama (two steps):
  python scripts/eval_rag.py --system graph-rag --judge claude-code --phase predict
  #   → Claude Code fills each row's "verdict" in reports/eval/rag/graph-rag__*.jsonl
  python scripts/eval_rag.py --system graph-rag --judge claude-code --phase report

  # Fine-tuned model (closed-book) on the lookup set:
  python scripts/eval_rag.py --system finetuned \
      --adapter runs/qlora-qwen2.5-3b-knowledge \
      --dataset data/article_lookup_golden.csv
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.eval import (  # noqa: E402
    aggregate, build_system, judge_phase_ollama, load_article_texts, load_gold_csv,
    make_judge, predict_phase, run_file, update_summary,
)
from legal_explainer.eval.runner import _read_jsonl_safe  # noqa: E402

DEFAULT_DATASETS = [
    "data/general_user_legal_questions.csv",
    "data/lawyer_llm_solution_questions.csv",
    "data/article_lookup_golden.csv",
]
DEFAULT_BASE_URL = {"graph-rag": "http://localhost:8000",
                    "bilingual-rag": "http://localhost:8100"}


def run_one(args, dataset: str) -> None:
    out_dir = Path(args.out_dir)
    path = run_file(out_dir, args.system, dataset)

    if args.phase in ("predict", "all"):
        rows = load_gold_csv(PROJECT_ROOT / dataset, limit=args.limit)
        print(f"\n=== {args.system} × {Path(dataset).stem}: {len(rows)} rows → {path} ===")
        kw = {"top_k": args.k}
        if args.system == "finetuned":
            if not args.adapter:
                raise SystemExit("--adapter is required for --system finetuned")
            kw["adapter_dir"] = str(PROJECT_ROOT / args.adapter)
            kw["base_model"] = args.base_model
            article_texts = load_article_texts()
        else:
            kw["base_url"] = args.base_url or DEFAULT_BASE_URL[args.system]
            article_texts = {}
        predict_phase(build_system(args.system, **kw), rows, article_texts, args.k, path)

    if args.phase in ("judge", "all"):
        if args.judge == "ollama":
            judge_phase_ollama(make_judge("ollama", model=args.judge_model,
                                          host=args.ollama_host), path)
        else:  # claude-code: scored in-session by editing each row's "verdict"
            print(f"\n[claude-code] open {path} and set each row's \"verdict\" to "
                  "{\"faithfulness\":0-1, \"answer_relevance\":0-1, \"context_precision\":0-1, "
                  "\"context_recall\":0-1, \"answer_correctness\":0-1} using its \"judge_prompt\".")
            if not any(r.get("verdict") for r in _read_jsonl_safe(path)) and args.phase == "all":
                print("  → no verdicts yet; run --phase report after grading.")
                return

    if args.phase in ("report", "all"):
        summary = aggregate(_read_jsonl_safe(path))
        update_summary(out_dir, args.system, dataset, args.k, args.judge, summary)
        o = summary["overall"]
        print(f"\n--- {args.system} × {Path(dataset).stem}  "
              f"({summary['n_rows']} rows, {summary['n_judged']} judged, "
              f"{summary['n_errors']} errors) ---")
        for m in ("faithfulness", "answer_relevance", "context_precision",
                  "context_recall", "answer_correctness", "retrieval_hit@k",
                  "citation_accuracy"):
            v = o.get(m)
            print(f"  {m:22s} {'—' if v is None else f'{v:.3f}'}")
        print(f"  → {out_dir/'SUMMARY.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--system", required=True,
                    choices=["graph-rag", "bilingual-rag", "finetuned"])
    ap.add_argument("--dataset", nargs="+", default=DEFAULT_DATASETS,
                    help="One or more gold CSVs (default: all three gold sets).")
    ap.add_argument("--judge", choices=["ollama", "claude-code"], default="ollama")
    ap.add_argument("--phase", choices=["predict", "judge", "report", "all"], default="all")
    ap.add_argument("--k", type=int, default=5, help="top-k retrieval depth")
    ap.add_argument("--limit", type=int, default=0, help="0 = all rows")
    ap.add_argument("--base-url", default="", help="RAG service URL (overrides default)")
    ap.add_argument("--adapter", default="", help="QLoRA adapter dir (finetuned only)")
    ap.add_argument("--base-model", default=None)
    ap.add_argument("--judge-model", default="qwen2.5:3b-instruct")
    ap.add_argument("--ollama-host", default="http://localhost:11434")
    ap.add_argument("--out-dir", default="reports/eval/rag")
    args = ap.parse_args()

    for dataset in args.dataset:
        run_one(args, dataset)


if __name__ == "__main__":
    main()
