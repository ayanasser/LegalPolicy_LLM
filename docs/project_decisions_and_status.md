# Project Decisions & Status

Single source of truth for the high-level decisions on the LegalPolicy_LLM project. Update this file whenever a non-obvious decision is locked in. Implementation details belong in code; this file records the *why* and *what's chosen*.

## Status snapshot (2026-05-04)

- **Scope:** bilingual (English + Arabic) legal explainer over the Egyptian Civil Code.
- **Hardware:** single RTX 3050 6 GB Laptop GPU on WSL2.
- **Active epic:** Epic 4 — Domain Adaptation (QLoRA fine-tuning).
- **Other epics:** scaffolded but not implemented. The decision to ship the tuned model is gated on Epic 3 (RAG) baseline numbers.

## Confirmed decisions

### Corpus
- Source of truth is [data/orig_data.json](../data/orig_data.json) — the Egyptian Civil Code, 1,093 bilingual articles (median 339 EN chars / 197 AR chars).

### Fine-tuning (Epic 4)

| Area | Decision |
|---|---|
| Method | QLoRA (4-bit NF4 base + LoRA via PEFT) |
| Languages | EN + AR, mixed monolingual (~50/50 EN→EN and AR→AR pairs) |
| Dataset | Hybrid template + Claude polish from `orig_data.json`; ~700-800 pairs; 85/15 train/val split |
| Refusal pairs | 7 EN + 7 AR hand-curated seeds in [src/legal_explainer/finetune/configs/refusal_seeds.yaml](../src/legal_explainer/finetune/configs/refusal_seeds.yaml), extendable |
| Synthesis + judge LLM | Anthropic Claude (Sonnet) |
| Eval | LLM-as-judge on a 100-question held-out set; base vs. tuned under identical RAG context |
| Deployment | Merge LoRA → GGUF q4_K_M → register with Ollama |
| Experiment tracking | TensorBoard (local, written to `runs/`) |

### Two training tracks

The local 6 GB GPU cannot host the original Epic 4 spec (Qwen 8B / 3B QLoRA) without OOM, so we ship two parallel paths from a shared dataset and pipeline:

- **Option B — primary, local: Qwen 2.5 1.5B Instruct.** Fits 6 GB comfortably. LoRA r=16, α=32, seq 1024, batch 1 × grad-accum 16, 3 epochs. Code: [src/legal_explainer/finetune/](../src/legal_explainer/finetune/) + [scripts/train.py](../scripts/train.py) / [scripts/build_dataset.py](../scripts/build_dataset.py) / [scripts/export_to_gguf.py](../scripts/export_to_gguf.py). Config: [configs/qlora_qwen1_5b_local.yaml](../src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml).
- **Stretch — cloud: Qwen 2.5 3B Instruct on Colab T4.** Original Epic 4 spec (LoRA r=32, α=64, seq 2048, batch 4 × grad-accum 8, 3 epochs). Code: [notebooks/qlora_qwen2_5_3b_colab.ipynb](../notebooks/qlora_qwen2_5_3b_colab.ipynb). The same hyperparameters are also captured in [configs/qlora_r32.yaml](../src/legal_explainer/finetune/configs/qlora_r32.yaml) so the local trainer can drive a cloud GPU later if needed.

The two tracks share dataset format, refusal seeds, synthesis prompts, and Modelfile template — adapters and merged GGUF artifacts are interchangeable.

## Open / deferred

- **Decision gate (Task 4.1):** the formal "do we ship the tuned model" decision is deferred until Epic 3 RAG baseline numbers exist. The pipeline is built ahead of that gate.
- **Epic 1 house-style template:** [config/prompts/system_prompt.yaml](../config/prompts/system_prompt.yaml) is still an empty scaffold. A v0 must be drafted before mass synthesis to keep Claude's polish step on-style.
- **Cost guardrail:** Anthropic budget for synthesis + LLM-judge eval ≈ $20–40 total.

## Change log

- **2026-05-04** — Epic 4 plan agreed. Local Option B (Qwen 1.5B) adopted as the primary track; Colab 3B notebook produced as the stretch track. Local code (dataset builder, trainer, merge/export, configs) implemented.
