"""Build the Stage-1 finetuning dataset for the LegalPolicy_LLM project.

Stage-1 data interventions (within PEFT, no method change):
  1. Per-article repetition: 8 question-phrasing variants per (article, language)
     instead of 2. Same article seen 4× more often → more content gradient signal.
  2. Cross-language parity: each chosen article is asked in BOTH EN and AR.
  3. Expanded refusal seeds: refusal_seeds_v2.yaml (50 entries, 12 categories)
     instead of refusal_seeds.yaml (14 entries).
  4. Contrastive pairs: ~30 examples that ask about non-existent article numbers
     and teach the model to refuse with a clear gap-acknowledgement, rather
     than fabricate content.

The polish cache is reused across question variants of the same (article,
language) — saves ~80% of API calls relative to per-instruction caching.

Usage:
    # Local dry-run (no API polish — uses raw article text as response):
    python scripts/build_dataset_stage1.py --no-polish --articles-per-lang 20

    # Full Stage-1 build (uses cached polishes from data/_polish_cache.json,
    # makes new calls only for any article not already cached):
    python scripts/build_dataset_stage1.py
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.dataset_builder import main  # noqa: E402


def _stage1_argv():
    """Inject Stage-1 defaults unless the user has already overridden them."""
    flags = sys.argv[1:]

    def has(flag: str) -> bool:
        return any(f == flag or f.startswith(flag + "=") for f in flags)

    additions = []
    if not has("--variants-per-article"):
        additions += ["--variants-per-article", "8"]
    if not has("--cross-lang-parity"):
        additions += ["--cross-lang-parity"]
    if not has("--n-contrastive"):
        additions += ["--n-contrastive", "60"]
    if not has("--refusal-seeds"):
        additions += ["--refusal-seeds", "refusal_seeds_v2.yaml"]
    if not has("--refusal-variants"):
        additions += ["--refusal-variants", "8"]
    if not has("--require-cached-polish"):
        additions += ["--require-cached-polish"]
    sys.argv = [sys.argv[0]] + additions + flags


if __name__ == "__main__":
    _stage1_argv()
    print("Stage-1 dataset build with: variants=8, bilingual parity, "
          "contrastive=30, refusal_seeds_v2.yaml")
    main()
