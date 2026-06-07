"""QLoRA fine-tuning via **Unsloth** (+ optional Weights & Biases logging).

Drop-in alternative to ``legal_explainer.finetune.train`` (raw PEFT/TRL) — reads
the *same* YAML config schema, so any existing config works here. Use this for
the bigger bases (Qwen2.5-3B) on the 6 GB GPU: Unsloth is ~2x faster and uses
roughly half the VRAM, which is what makes 3B QLoRA comfortable at seq 1536.

If the config has a ``wandb:`` section (and `wandb` is installed and not
disabled via the WANDB_MODE / WANDB_DISABLED env vars), training is logged to
W&B; otherwise it falls back to TensorBoard under ``<output_dir>/runs``.

Usage:
    python scripts/train_unsloth.py --config src/legal_explainer/finetune/configs/qlora_qwen3b_raft.yaml

Requires:  pip install "unsloth" wandb        (unsloth pulls a compatible
torch/triton; on WSL2 + an Ampere GPU like the RTX 3050 this works out of the box)
"""
from __future__ import annotations

# Unsloth must be imported before torch/transformers so its kernel patches apply.
from unsloth import FastLanguageModel  # noqa: E402  (import order matters)

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
from pathlib import Path  # noqa: E402

import torch  # noqa: E402
import yaml  # noqa: E402
from datasets import load_dataset  # noqa: E402
from transformers import EarlyStoppingCallback  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

try:  # optional
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if load_dotenv:
    load_dotenv(PROJECT_ROOT / ".env")


def load_config(path: Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _wandb_enabled(cfg: dict) -> bool:
    if not cfg.get("wandb"):
        return False
    mode = (os.environ.get("WANDB_MODE") or "").lower()
    if os.environ.get("WANDB_DISABLED", "").lower() in ("true", "1") or mode == "disabled":
        return False
    try:
        import wandb  # noqa: F401
    except ImportError:
        print("[train_unsloth] wandb not installed — falling back to tensorboard.")
        return False
    return True


def build_chat_dataset(train_jsonl: Path, val_jsonl: Path, tokenizer):
    ds = load_dataset("json", data_files={"train": str(train_jsonl), "validation": str(val_jsonl)})

    def fmt(ex):
        return {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False,
                                                      add_generation_prompt=False)}

    drop = [c for c in ds["train"].column_names if c != "messages"]
    return ds.map(fmt, remove_columns=drop)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="Tiny smoke test: a handful of train samples, ~3 steps, no eval/save/early-stopping. "
                         "Use to check the whole pipeline (model load, LoRA, data, trainer) end to end.")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Cap optimisation steps (overrides num_train_epochs). Set automatically by --dry-run.")
    ap.add_argument("--max-train-samples", type=int, default=None,
                    help="Use only the first N training examples (debug / dry run).")
    ap.add_argument("--max-eval-samples", type=int, default=None,
                    help="Use only the first N validation examples (debug / dry run).")
    ap.add_argument("--resume", nargs="?", const=True, default=None,
                    help="Resume from a checkpoint. With no value, HF Trainer auto-detects the "
                         "latest checkpoint in output_dir. Pass a path to resume from a specific dir.")
    args = ap.parse_args()
    if args.dry_run:
        args.max_steps = args.max_steps or 3
        args.max_train_samples = args.max_train_samples or 16
        args.max_eval_samples = args.max_eval_samples or 8

    cfg = load_config(args.config)
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    t, lora = cfg["training"], cfg["lora"]

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; QLoRA requires a GPU.")
    p = torch.cuda.get_device_properties(0)
    print(f"Base model : {cfg['base_model']}")
    print(f"Output dir : {output_dir}")
    print(f"GPU        : {p.name}  VRAM={p.total_memory/1e9:.1f} GB  bf16={torch.cuda.is_bf16_supported()}")
    print("Engine     : Unsloth")

    # --- model + LoRA (Unsloth) -------------------------------------------
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=cfg["base_model"],
        max_seq_length=t["max_seq_length"],
        dtype=None,                       # auto: bf16 on Ampere, fp16 otherwise
        load_in_4bit=bool(cfg["quantization"]["load_in_4bit"]),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = FastLanguageModel.get_peft_model(
        model,
        r=lora["r"],
        target_modules=lora["target_modules"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        bias=lora["bias"],
        use_gradient_checkpointing="unsloth" if t.get("gradient_checkpointing", True) else False,
        random_state=t["seed"],
    )

    # --- data -------------------------------------------------------------
    train_jsonl = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_jsonl = PROJECT_ROOT / cfg["data"]["val_jsonl"]
    if not train_jsonl.exists() or not val_jsonl.exists():
        raise FileNotFoundError(f"Training data not found: {train_jsonl} or {val_jsonl}.")
    ds = build_chat_dataset(train_jsonl, val_jsonl, tokenizer)
    if args.max_train_samples:
        ds["train"] = ds["train"].select(range(min(args.max_train_samples, len(ds["train"]))))
    if args.max_eval_samples:
        ds["validation"] = ds["validation"].select(range(min(args.max_eval_samples, len(ds["validation"]))))
    print(f"Data       : train={len(ds['train'])}  val={len(ds['validation'])}"
          + (f"   [DRY RUN — max_steps={args.max_steps}]" if args.dry_run else ""))

    # --- experiment tracking (optional) -----------------------------------
    # Log to W&B AND TensorBoard whenever each is available. The two are
    # independent and additive (no conflict in HF Trainer), so the user gets
    # both: live W&B dashboard + local TB events under <output_dir>/runs/ that
    # `tensorboard --logdir <output_dir>/runs` can serve.
    run_name = cfg.get("adapter_name") or output_dir.name
    report_to: list[str] = []
    if _wandb_enabled(cfg):
        import wandb
        wb = cfg["wandb"] or {}
        os.environ.setdefault("WANDB_PROJECT", wb.get("project", "legalpolicy"))
        if wb.get("mode"):
            os.environ.setdefault("WANDB_MODE", str(wb["mode"]))
        run_name = wb.get("run_name") or run_name
        wandb.init(project=wb.get("project", "legalpolicy"), name=run_name,
                   job_type="train", config=cfg)
        report_to.append("wandb")
        print(f"Tracking   : W&B project={wb.get('project', 'legalpolicy')} run={run_name}")
    try:
        import tensorboard  # noqa: F401
        report_to.append("tensorboard")
        print(f"Tracking   : TensorBoard -> {output_dir / 'runs'}"
              f"   (serve: tensorboard --logdir {output_dir / 'runs'})")
    except ImportError:
        if not report_to:
            print("Tracking   : none (no wandb section / tensorboard not installed)")
    if not report_to:
        report_to = ["none"]

    # --- trainer ----------------------------------------------------------
    bf16_ok = bool(t["bf16"]) and torch.cuda.is_bf16_supported()
    eval_strategy = "no" if args.dry_run else t["eval_strategy"]
    save_strategy = "no" if args.dry_run else t["save_strategy"]
    load_best = False if args.dry_run else t["load_best_model_at_end"]
    callbacks = ([] if args.dry_run
                 else [EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])])
    sft = SFTConfig(
        output_dir=str(output_dir),
        run_name=run_name,
        num_train_epochs=t["num_train_epochs"],
        max_steps=(args.max_steps if args.max_steps else -1),
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        bf16=bf16_ok,
        fp16=not bf16_ok,
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        eval_strategy=eval_strategy,
        eval_steps=t["eval_steps"],
        save_strategy=save_strategy,
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=load_best,
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        report_to=report_to,
        logging_dir=str(output_dir / "runs"),
        max_length=t["max_seq_length"],
        dataset_text_field="text",
        packing=t["packing"],
        seed=t["seed"],
    )
    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    if args.resume is not None:
        print(f"Resume     : {args.resume!r} (HF Trainer will load optimizer/scheduler/rng + LoRA weights)")
    trainer.train(resume_from_checkpoint=args.resume)
    trainer.save_model(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))
    (output_dir / "training_config_used.json").write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    if args.dry_run:
        print(f"DRY RUN OK — pipeline ran end to end ({args.max_steps} steps); a (throwaway-quality) "
              f"adapter was written to {output_dir} so the RAG `ask`/`eval` path can be smoke-tested. "
              f"Re-run without --dry-run for a real run.")
    else:
        print(f"Adapter saved to {output_dir}")

    if report_to == ["wandb"]:
        import wandb
        wandb.finish()


if __name__ == "__main__":
    main()
