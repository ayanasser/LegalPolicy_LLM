"""RAFT-RAG — a small, hand-rolled hybrid (BM25 + dense) retriever over the
Egyptian Civil Code, wired to the RAFT-trained LoRA adapter so the model answers
*grounded in the retrieved article text*, using the **exact** context-block
format the adapter was fine-tuned on (``dataset_builder._format_context_block``).

This is the RAFT track's RAG and lives entirely under ``finetune/`` on purpose:
the standalone product RAG (root ``scripts/``, owned by a teammate) is separate
and must not be touched from here.

CLI:  ``python -m legal_explainer.finetune.raft_rag <command>``
    build-index   build & persist the article index (BM25 + optional dense)
    retrieve      show what the retriever returns for a query (no LLM, no GPU)
    ask           full pipeline: retrieve -> RAFT prompt -> generate
    eval          score closed-book vs RAG vs oracle on a question set (+wandb)

Heavy modules (``infer``, ``eval``) pull in torch/transformers and are imported
lazily — importing this package only loads the light retrieval pieces.
"""
from .index import ArticleEntry, ArticleIndex, build_index, detect_lang, tokenize
from .prompt import build_prompt
from .retriever import HybridRetriever, RetrievalResult, Retriever

__all__ = [
    "ArticleEntry", "ArticleIndex", "build_index", "detect_lang", "tokenize",
    "HybridRetriever", "RetrievalResult", "Retriever",
    "build_prompt",
]
