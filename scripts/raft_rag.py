"""Launcher for the RAFT-track RAG CLI (logic lives in
legal_explainer.finetune.raft_rag — kept under finetune/ on purpose; the
root-level product RAG is separate and owned by a teammate).

    python scripts/raft_rag.py build-index [--no-dense]
    python scripts/raft_rag.py retrieve "Explain Article 775 simply"
    python scripts/raft_rag.py ask "Explain Article 775 simply" [--show-prompt]
    python scripts/raft_rag.py eval --set data/qa_pairs_raft_val.jsonl [--wandb]
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.raft_rag.__main__ import main  # noqa: E402

if __name__ == "__main__":
    main()
