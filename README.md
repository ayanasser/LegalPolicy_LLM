# LegalPolicy_LLM

A bilingual (English + Arabic) legal explainer over the **Egyptian Civil Code**. The project is organized into 8 epics covering scope, corpus, RAG, fine-tuning, tooling, multi-agent collaboration, evaluation, and observability — see [docs/epics_tasks.md](docs/epics_tasks.md) for the full plan and [docs/project_decisions_and_status.md](docs/project_decisions_and_status.md) for the current decisions log.

This README is scoped to the **fine-tuning epic** (QLoRA domain adaptation). Other epics will get their own setup notes as they come online.

---

## What the fine-tuning epic does

Adapts a small open Qwen 2.5 instruct model to bilingual explanations of Egyptian Civil Code articles, in the project's house style (definition-first, structured, plain language, with a refusal/disclaimer policy). It does this with **QLoRA** — 4-bit quantized base + LoRA adapters — so it runs on a consumer GPU.

Two parallel training tracks are supported from a shared dataset and pipeline:

| Track | Base model | Where it runs | Recipe |
|---|---|---|---|
| **Local (primary)** | Qwen 2.5 **1.5B** Instruct | Single 6 GB GPU on WSL2 / Linux | [configs/qlora_qwen1_5b_local.yaml](src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml) |
| **Cloud (stretch)** | Qwen 2.5 **3B** Instruct | Google Colab T4 (16 GB) | [notebooks/qlora_qwen2_5_3b_colab.ipynb](notebooks/qlora_qwen2_5_3b_colab.ipynb) |

Both produce a LoRA adapter, optionally merged into the base and exported to GGUF (q4_K_M) for serving via Ollama.

---

## Prerequisites (local track)

- Linux or WSL2 (this project is developed on WSL2 / Ubuntu)
- NVIDIA GPU with **≥ 6 GB VRAM** and CUDA 12.x driver. Verify with `nvidia-smi`.
- **Miniconda or Anaconda** installed.
- A local checkout of [llama.cpp](https://github.com/ggerganov/llama.cpp) — only needed for the optional GGUF export step.
- An **Anthropic API key** if you want Claude to polish the synthesized training pairs into the house style. Set `ANTHROPIC_API_KEY` in your shell (or in a `.env` file). Without this, the dataset builder falls back to raw article text as the response (lower style quality, free).

For the cloud track you only need a Google account; everything else is set up inside the Colab notebook.

---

## Setup (conda environment)

```bash
# 1) clone and enter the repo
git clone <your-repo-url> LegalPolicy_LLM
cd LegalPolicy_LLM

# 2) create and activate the conda env (Python 3.11 is the supported version)
conda create -n legalpolicy python=3.11 -y
conda activate legalpolicy

# 3) install PyTorch with the right CUDA build first (separate channel — pip will not pick the right wheel by itself)
pip install --index-url https://download.pytorch.org/whl/cu121 \
    "torch>=2.4"

# 4) install the rest of the project's local requirements (includes the fine-tuning stack)
pip install -r requirements-local.txt

# 5) sanity check
python -c "import torch; print('cuda?', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
```

If step 5 prints `cuda? True` and your GPU name, the environment is ready.

---

## Configure secrets

```bash
# in the repo root, write a .env file (gitignored) — or just export inline
echo 'ANTHROPIC_API_KEY=sk-ant-...' >> .env
export ANTHROPIC_API_KEY=sk-ant-...
```

The fine-tuning code reads `ANTHROPIC_API_KEY` from the environment.

---

## Run the fine-tuning pipeline (local track)

The local pipeline has three commands, run in order. Each is idempotent and can be re-run without losing prior work.

### 1. Build the training dataset (~700–800 pairs)

Generates instruction-response pairs from `data/orig_data.json` (Egyptian Civil Code, 1,093 bilingual articles), polishes them via Claude, injects refusal seeds, and writes the train/val splits.

```bash
python scripts/build_dataset.py \
    --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml
```

Useful flags:
- `--no-polish` — skip the Claude polish step (free, lower style quality; good for smoke-testing).
- `--articles-per-lang 100` — smaller dataset for a fast pipeline check.
- `--polish-model claude-sonnet-4-6` — change the polish model.

Outputs:
- `data/qa_pairs.jsonl` — training split
- `data/qa_pairs_val.jsonl` — validation split
- `data/_polish_cache.json` — per-pair Claude cache so re-runs are cheap

### 2. Train the QLoRA adapter

Loads the base model in 4-bit, applies LoRA, runs SFT with TensorBoard logging.

```bash
python scripts/train.py \
    --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml
```

Outputs:
- `runs/qlora-qwen2.5-1.5b-v1/` — adapter checkpoints, tokenizer, training config snapshot
- `runs/qlora-qwen2.5-1.5b-v1/runs/` — TensorBoard event files

Watch training in real time:
```bash
tensorboard --logdir runs/qlora-qwen2.5-1.5b-v1/runs
```

### 3. (Optional) Merge + export to GGUF for Ollama

After eval looks good. Requires a local llama.cpp checkout that has been built:

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp ~/llama.cpp
cd ~/llama.cpp && cmake -B build && cmake --build build --target llama-quantize -j
cd -

python scripts/export_to_gguf.py \
    --adapter   runs/qlora-qwen2.5-1.5b-v1 \
    --base      Qwen/Qwen2.5-1.5B-Instruct \
    --llama-cpp ~/llama.cpp \
    --out-name  legalpolicy-qwen1.5b

# then register with Ollama
cd artifacts
ollama create legalpolicy-qwen1.5b -f legalpolicy-qwen1.5b.Modelfile
ollama run legalpolicy-qwen1.5b
```

---

## Run the fine-tuning pipeline (cloud track)

Open [notebooks/qlora_qwen2_5_3b_colab.ipynb](notebooks/qlora_qwen2_5_3b_colab.ipynb) in Google Colab, set runtime to **T4 GPU**, add `ANTHROPIC_API_KEY` to Colab Secrets, and run cells top-to-bottom. Full instructions are in [notebooks/README.md](notebooks/README.md).

The notebook is self-contained (uploads `orig_data.json`, builds the dataset inside the notebook, trains, optionally merges and exports to GGUF). Resulting artifacts can be downloaded and registered with Ollama on your laptop using the same Modelfile pattern as the local track.

---

## Project layout (fine-tuning slice)

```
LegalPolicy_LLM/
├── data/
│   ├── orig_data.json                 # bilingual Egyptian Civil Code (1,093 articles)
│   ├── qa_pairs.jsonl                 # generated by build_dataset.py
│   └── qa_pairs_val.jsonl
├── src/legal_explainer/finetune/
│   ├── dataset_builder.py             # templates → Claude polish → refusal injection → split
│   ├── train.py                       # QLoRA training (TRL SFTTrainer)
│   ├── merge_export.py                # adapter → merged HF checkpoint
│   └── configs/
│       ├── qlora_qwen1_5b_local.yaml  # 6 GB-safe local recipe
│       ├── qlora_r32.yaml             # original Epic 4 spec (cloud / 3B)
│       ├── synthesis_prompts.yaml     # Claude polish prompts
│       └── refusal_seeds.yaml         # 7 EN + 7 AR refusal pairs
├── scripts/
│   ├── build_dataset.py               # CLI wrapper
│   ├── train.py                       # CLI wrapper
│   └── export_to_gguf.py              # merge + llama.cpp convert + quantize + Modelfile
├── notebooks/
│   └── qlora_qwen2_5_3b_colab.ipynb   # cloud track
├── runs/                              # training outputs (gitignored)
├── artifacts/                         # merged HF checkpoint + GGUF + Modelfile (gitignored)
├── requirements.txt                   # core deps (RAG, app, eval)
├── requirements-local.txt             # everything in requirements.txt + local-runtime + fine-tuning stack
└── docs/
    ├── epics_tasks.md                 # full 8-epic plan
    └── project_decisions_and_status.md
```

---

## Troubleshooting

- **`bitsandbytes` import error on Linux/WSL2** — usually a CUDA version mismatch. Confirm `python -m bitsandbytes` runs cleanly; if not, reinstall after fixing the torch CUDA build.
- **OOM during training on the local track** — reduce `max_seq_length` to 768, set `per_device_train_batch_size: 1` (already the default), and bump `gradient_accumulation_steps`. If still OOM, switch to the cloud track.
- **Claude polish takes too long / costs too much** — start with `--articles-per-lang 100` (≈ 200 pairs total) and `--no-polish` to validate the pipeline before paying for the full run. The cache makes a second polished run cheap.
- **`ollama create` rejects the GGUF** — confirm the Modelfile's `FROM ./<gguf>` path is relative to the directory you run `ollama create` from.
