"""RAGAS-style evaluation for the Legal Explainer RAG / fine-tuned systems.

Metrics
-------
LLM-judged (0–1, via `judge.py`):
    faithfulness · answer_relevance · context_precision · context_recall ·
    answer_correctness
Deterministic (via `scores.py`):
    citation_accuracy · has_citation · retrieval_hit@k · retrieval_mrr ·
    context_precision@k · token_overlap_gold

Judge backends: `ollama` (automated) or `claude-code` (file handoff).
CLI entry point: `scripts/eval_rag.py`.
"""
from .datasets import GoldRow, load_article_texts, load_gold_csv
from .judge import ClaudeCodeJudge, OllamaJudge, make_judge, make_task, parse_verdict
from .runner import aggregate, judge_phase_ollama, predict_phase, run_file, update_summary
from .scores import deterministic_metrics
from .systems import BilingualRAGSystem, FinetuneSystem, GraphRAGSystem, build_system

__all__ = [
    "GoldRow", "load_gold_csv", "load_article_texts",
    "make_judge", "make_task", "parse_verdict", "OllamaJudge", "ClaudeCodeJudge",
    "deterministic_metrics",
    "GraphRAGSystem", "BilingualRAGSystem", "FinetuneSystem", "build_system",
    "predict_phase", "judge_phase_ollama", "aggregate", "run_file", "update_summary",
]
