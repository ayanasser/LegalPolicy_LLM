"""Build the knowledge-injection dataset (Stage A of the two-stage PEFT recipe).

Thin CLI wrapper around legal_explainer.finetune.knowledge_builder. Generates a
plain, varied, no-API dataset from data/orig_data.json whose only purpose is to
imprint the verbatim Egyptian Civil Code text into the LoRA weights — see the
module docstring for the rationale and task families.

Usage:
    python scripts/build_dataset_knowledge.py \
        --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_knowledge.yaml

    # smoke test:
    python scripts/build_dataset_knowledge.py --config <cfg> --max-articles 25
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.knowledge_builder import main  # noqa: E402

if __name__ == "__main__":
    main()
