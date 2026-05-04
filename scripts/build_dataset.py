"""Build the QLoRA training dataset via legal_explainer.finetune.dataset_builder."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.dataset_builder import main  # noqa: E402

if __name__ == "__main__":
    main()
