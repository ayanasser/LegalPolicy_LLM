"""QLoRA fine-tuning entrypoint for the LegalPolicy_LLM project.

Reads a YAML config, loads the base model in 4-bit, applies LoRA adapters,
trains via TRL's SFTTrainer, and saves the adapter to <output_dir>.

Usage:
    python scripts/train.py --config src/legal_explainer/finetune/configs/qlora_qwen1_5b_local.yaml
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    EarlyStoppingCallback,
)
from trl import SFTConfig, SFTTrainer

PROJECT_ROOT = Path(__file__).resolve().parents[3]


def load_config(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def build_tokenizer(model_id: str):
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def build_model(cfg: dict):
    quant = cfg["quantization"]
    bnb = BitsAndBytesConfig(
        load_in_4bit=quant["load_in_4bit"],
        bnb_4bit_quant_type=quant["bnb_4bit_quant_type"],
        bnb_4bit_use_double_quant=quant["bnb_4bit_use_double_quant"],
        bnb_4bit_compute_dtype=getattr(torch, quant["bnb_4bit_compute_dtype"]),
    )
    model = AutoModelForCausalLM.from_pretrained(
        cfg["base_model"],
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_cfg = LoraConfig(
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        bias=cfg["lora"]["bias"],
        task_type=cfg["lora"]["task_type"],
        target_modules=cfg["lora"]["target_modules"],
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    return model


def load_chat_dataset(train_jsonl: Path, val_jsonl: Path, tokenizer):
    ds = load_dataset("json", data_files={
        "train": str(train_jsonl),
        "validation": str(val_jsonl),
    })

    def fmt(ex):
        text = tokenizer.apply_chat_template(
            ex["messages"], tokenize=False, add_generation_prompt=False,
        )
        return {"text": text}

    drop_cols = [c for c in ds["train"].column_names if c != "messages"]
    ds = ds.map(fmt, remove_columns=drop_cols)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, type=Path)
    args = ap.parse_args()

    cfg = load_config(args.config)
    output_dir = PROJECT_ROOT / cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Base model: {cfg['base_model']}")
    print(f"Output dir: {output_dir}")
    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available; QLoRA requires a GPU.")
    p = torch.cuda.get_device_properties(0)
    print(f"GPU: {p.name}  VRAM={p.total_memory/1e9:.1f} GB  bf16={torch.cuda.is_bf16_supported()}")

    tokenizer = build_tokenizer(cfg["base_model"])
    model = build_model(cfg)

    train_jsonl = PROJECT_ROOT / cfg["data"]["train_jsonl"]
    val_jsonl = PROJECT_ROOT / cfg["data"]["val_jsonl"]
    if not train_jsonl.exists() or not val_jsonl.exists():
        raise FileNotFoundError(
            f"Training data not found: {train_jsonl} or {val_jsonl}. "
            "Run scripts/build_dataset.py first."
        )
    ds = load_chat_dataset(train_jsonl, val_jsonl, tokenizer)

    t = cfg["training"]
    sft = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=t["num_train_epochs"],
        per_device_train_batch_size=t["per_device_train_batch_size"],
        per_device_eval_batch_size=t["per_device_eval_batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        lr_scheduler_type=t["lr_scheduler_type"],
        warmup_ratio=t["warmup_ratio"],
        bf16=t["bf16"],
        gradient_checkpointing=t["gradient_checkpointing"],
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim=t["optim"],
        logging_steps=t["logging_steps"],
        eval_strategy=t["eval_strategy"],
        eval_steps=t["eval_steps"],
        save_strategy=t["save_strategy"],
        save_total_limit=t["save_total_limit"],
        load_best_model_at_end=t["load_best_model_at_end"],
        metric_for_best_model=t["metric_for_best_model"],
        greater_is_better=t["greater_is_better"],
        report_to=["tensorboard"],
        logging_dir=str(output_dir / "runs"),
        max_seq_length=t["max_seq_length"],
        dataset_text_field="text",
        packing=t["packing"],
        seed=t["seed"],
    )

    trainer = SFTTrainer(
        model=model,
        args=sft,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=t["early_stopping_patience"])],
    )

    trainer.train()
    trainer.save_model(str(output_dir))
    (output_dir / "training_config_used.json").write_text(
        json.dumps(cfg, indent=2), encoding="utf-8",
    )
    print(f"Adapter saved to {output_dir}")


if __name__ == "__main__":
    main()
