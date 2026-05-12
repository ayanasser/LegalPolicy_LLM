"""CLI for the RAFT-track RAG.

    python -m legal_explainer.finetune.raft_rag build-index [--config ...] [--no-dense]
    python -m legal_explainer.finetune.raft_rag retrieve  "<query>" [--config ...] [-k N]
    python -m legal_explainer.finetune.raft_rag ask        "<question>" [--config ...] [--mode rag|closed] [--show-prompt]
    python -m legal_explainer.finetune.raft_rag eval       --set <jsonl> [--config ...] [--modes closed,rag,oracle] [--wandb]

`build-index`, `retrieve` and `eval` work without a GPU; `ask` / `eval` load the
RAFT adapter (4-bit) for generation.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

# .../src/legal_explainer/finetune/raft_rag/__main__.py  ->  .../LegalPolicy_LLM
PKG_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(PKG_ROOT / "src"))

from legal_explainer.finetune.raft_rag.index import (  # noqa: E402
    DEFAULT_ARTIFACTS, DEFAULT_CORPUS, build_index,
)
from legal_explainer.finetune.raft_rag.retriever import HybridRetriever  # noqa: E402

DEFAULT_CONFIG = PKG_ROOT / "src" / "legal_explainer" / "finetune" / "configs" / "raft_rag.yaml"


def _load_cfg(path) -> dict:
    p = Path(path)
    return (yaml.safe_load(p.read_text(encoding="utf-8")) or {}) if p.exists() else {}


def _abs(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else PKG_ROOT / p


def cmd_build_index(args):
    cfg = _load_cfg(args.config)
    emb = cfg.get("embedder", {}) or {}
    idx = build_index(
        corpus_path=_abs(args.corpus or cfg.get("corpus", DEFAULT_CORPUS)),
        embedder=args.embedder or emb.get("model", "intfloat/multilingual-e5-small"),
        query_prefix=emb.get("query_prefix", "query: "),
        passage_prefix=emb.get("passage_prefix", "passage: "),
        device=args.device or emb.get("device", "cpu"),
        batch_size=emb.get("batch_size", 32),
        with_dense=not args.no_dense,
    )
    out = _abs(args.out_dir or cfg.get("artifacts_dir", DEFAULT_ARTIFACTS))
    idx.save(out)
    dshape = ("%dx%d" % idx.dense.shape) if idx.dense is not None else "disabled"
    print(f"Indexed {len(idx.entries)} articles -> {out}")
    print(f"  bm25: yes   dense: {dshape}   embedder: {idx.meta.get('embedder')}")


def cmd_retrieve(args):
    cfg = _load_cfg(args.config)
    rc = cfg.get("retrieval", {}) or {}
    retr = HybridRetriever.load(_abs(cfg.get("artifacts_dir", DEFAULT_ARTIFACTS)),
                                rrf_k=rc.get("rrf_k", 60), candidate_pool=rc.get("candidate_pool", 20))
    r = retr.retrieve(args.query, lang=args.lang, k=args.k, n_distractors=args.n_distractors)
    print(f"query: {args.query!r}   lang={args.lang or 'auto'}   explicit_id={r.explicit_id}")
    print("oracle:")
    for e in r.oracle:
        print(f"  {e.key}  topic_en={e.topic_en!r}  :: {(e.english or e.arabic)[:120]}")
    print("distractors:")
    for e in r.distractors:
        print(f"  {e.key}  :: {(e.english or e.arabic)[:80]}")
    print("top fused:", json.dumps(r.scores.get("fused_top"), ensure_ascii=False))


def cmd_ask(args):
    from legal_explainer.finetune.raft_rag.infer import RaftRagPipeline
    cfg = _load_cfg(args.config)
    pipe = RaftRagPipeline.from_config(cfg, load_model=True)
    rec = pipe.ask(args.question, mode=args.mode)
    print(json.dumps({k: v for k, v in rec.items() if k != "prompt"}, ensure_ascii=False, indent=2))
    if args.show_prompt:
        print("\n--- prompt sent to the model ---\n" + (rec.get("prompt") or ""))


def cmd_eval(args):
    from legal_explainer.finetune.raft_rag.eval import evaluate
    cfg = _load_cfg(args.config)
    set_path = _abs(args.set)
    rows = [json.loads(l) for l in set_path.read_text(encoding="utf-8").splitlines() if l.strip()]
    wb = None
    if args.wandb:
        import wandb
        wb = wandb.init(
            project=args.wandb_project or (cfg.get("eval", {}) or {}).get("wandb_project", "legalpolicy-raft-rag"),
            job_type="raft-rag-eval",
            config={"set": str(set_path), "modes": args.modes.split(","),
                    "adapter": (cfg.get("generation", {}) or {}).get("adapter_dir")},
        )
    out = evaluate(cfg, rows, modes=tuple(args.modes.split(",")), limit=args.limit, wandb_run=wb)
    out_path = _abs(args.out or (cfg.get("eval", {}) or {}).get("out_path", "reports/eval/raft_rag_eval.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n=== summary ===")
    for m, d in out["summary"].items():
        print(f"  {m:7s}: {json.dumps(d, ensure_ascii=False)}")
    print(f"-> {out_path}")
    if wb is not None:
        wb.finish()


def main():
    ap = argparse.ArgumentParser(prog="python -m legal_explainer.finetune.raft_rag")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("build-index", help="build & persist the article index (BM25 + optional dense)")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--corpus")
    p.add_argument("--out-dir")
    p.add_argument("--embedder")
    p.add_argument("--device")
    p.add_argument("--no-dense", action="store_true", help="BM25-only (skip the embedding model download)")
    p.set_defaults(fn=cmd_build_index)

    p = sub.add_parser("retrieve", help="show retriever output for a query (no LLM, no GPU)")
    p.add_argument("query")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--lang", choices=["en", "ar"])
    p.add_argument("-k", type=int, default=1)
    p.add_argument("--n-distractors", type=int, default=1)
    p.set_defaults(fn=cmd_retrieve)

    p = sub.add_parser("ask", help="full pipeline: retrieve -> RAFT prompt -> generate")
    p.add_argument("question")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--mode", choices=["rag", "closed"], default="rag")
    p.add_argument("--show-prompt", action="store_true")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("eval", help="score closed-book vs RAG vs oracle on a question set")
    p.add_argument("--config", default=str(DEFAULT_CONFIG))
    p.add_argument("--set", required=True, help="JSONL of questions (qa_pairs format or flat {question,...})")
    p.add_argument("--modes", default="closed,rag,oracle")
    p.add_argument("--limit", type=int)
    p.add_argument("--out")
    p.add_argument("--wandb", action="store_true")
    p.add_argument("--wandb-project")
    p.set_defaults(fn=cmd_eval)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
