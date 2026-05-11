# notebooks/

Jupyter notebooks used for work that needs cloud GPU or interactive exploration. The local Python package under [../src/legal_explainer/](../src/legal_explainer/) and the CLI in [../scripts/](../scripts/) remain the canonical place for code that runs in production or in CI; notebooks here are for experiments that do not fit on the local 6 GB GPU.

## Contents

| File | Purpose | Where to run |
|---|---|---|
| [qlora_qwen2_5_3b_colab.ipynb](qlora_qwen2_5_3b_colab.ipynb) | End-to-end QLoRA fine-tuning of Qwen 2.5 3B Instruct on the Egyptian Civil Code (EN + AR), per Epic 4. Builds the training dataset from [../data/orig_data.json](../data/orig_data.json), trains, evaluates with sample generations, and optionally merges + exports to GGUF for Ollama. | Google Colab (T4 16 GB or better) |

## When to reach for a notebook here vs. the local code

- **Use the notebook** when training or evaluating the **3B** (or larger) variant. The 6 GB local GPU cannot hold a 3B QLoRA without painful tradeoffs.
- **Use [../src/legal_explainer/finetune/](../src/legal_explainer/finetune/) and [../scripts/train.py](../scripts/train.py)** when training the **1.5B** local variant — the supported path on the development laptop.
- Both paths produce a LoRA adapter compatible with the same merge + GGUF export step. The notebook and the local package share dataset format (chat-template JSONL) so artifacts are interchangeable.

## Running `qlora_qwen2_5_3b_colab.ipynb`

1. Open the notebook in Colab and set **Runtime → Change runtime type → T4 GPU** (L4 / A100 will train faster).
2. Add Colab Secrets:
   - `ANTHROPIC_API_KEY` — required if `USE_CLAUDE_POLISH = True` (recommended; produces house-style training responses).
   - `HF_TOKEN` — only if you flip `PUSH_TO_HUB = True` to publish the adapter.
3. Have [../data/orig_data.json](../data/orig_data.json) reachable. Step 3 in the notebook supports three modes: file upload, `git clone` from your GitHub fork, or copy from Google Drive.
4. Run cells top-to-bottom. Drive mount in Step 6 lets training survive Colab disconnects — checkpoints land under `/content/drive/MyDrive/legalpolicy_qlora/`.
5. Wall-clock on T4: dataset build ~10–25 min (with Claude polish for ~700 pairs), training ~45–90 min, optional merge + GGUF ~10 min.

## Cost estimate

- **Compute:** Colab Free is sufficient for one full training run; Colab Pro (~$10/mo) reduces disconnects.
- **Anthropic API:** ~$15–30 for ~700 polished pairs at Sonnet rates (~700 output tokens / pair). Set `USE_CLAUDE_POLISH = False` to skip and use raw article text instead — quality drops but cost is zero.

## Conventions

- All experiment-time knobs are concentrated in the **Step 2 — Configuration** cell. Edit there, not scattered through downstream cells.
- Adapters are versioned by `ADAPTER_NAME` (default `qlora-qwen2.5-3b-v1`). Bump the suffix on each meaningful run so old checkpoints are not overwritten.
- The notebook does not run automated eval (Task 4.7); after training, download the adapter or GGUF and run [../scripts/run_eval.py](../scripts/run_eval.py) locally against the held-out evaluation set.
