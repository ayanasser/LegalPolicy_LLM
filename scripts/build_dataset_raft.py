"""Build the Stage-4 (PEFT-RAFT) finetuning dataset for LegalPolicy_LLM.

Same data interventions as Stage 1 (8 question variants per article, cross-
language parity, refusal_seeds_v2, 8x refusal augmentation, contrastive
"article-doesn't-exist" pairs) PLUS the RAFT transformation: each question is
preceded by a context block containing the article it is about (the "oracle")
and one distractor article. The gold answer is unchanged.

Why this is still PEFT: only the LoRA adapter weights update; the 4-bit base
stays frozen. RAFT changes what the adapter is trained to do (locate + rephrase
the article in context) — and, as a side effect, exposes the adapter to every
article's text many times, which is the strongest content-learning signal
available without enlarging the base model.

Usage:
    # Local dry-run (no API polish):
    python scripts/build_dataset_raft.py --no-polish --articles-per-lang 20

    # Full build (uses cached polishes; zero API cost on the cached path):
    python scripts/build_dataset_raft.py \
        --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_raft.yaml
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from legal_explainer.finetune.dataset_builder import main  # noqa: E402


def _raft_argv():
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
    if not has("--raft"):
        additions += ["--raft"]
    if not has("--raft-distractors"):
        additions += ["--raft-distractors", "1"]
    sys.argv = [sys.argv[0]] + additions + flags


if __name__ == "__main__":
    _raft_argv()
    print("Stage-4 PEFT-RAFT dataset build: variants=8, bilingual parity, "
          "contrastive=60, refusal_seeds_v2.yaml x8, RAFT context (oracle + 1 distractor)")
    main()
