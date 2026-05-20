"""Train a QLoRA adapter via Unsloth (+ optional W&B) — wrapper around
legal_explainer.finetune.train_unsloth.

    python scripts/train_unsloth.py --config src/legal_explainer/finetune/configs/qlora_qwen3b_raft.yaml
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.train_unsloth import main  # noqa: E402

if __name__ == "__main__":
    main()
