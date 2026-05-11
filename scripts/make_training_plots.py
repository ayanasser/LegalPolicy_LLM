"""Generate training-report plots from a trainer_state.json.

Reads runs/<run>/checkpoint-N/trainer_state.json and writes:
  reports/training/loss_curve.png
  reports/training/accuracy_curve.png
  reports/training/lr_schedule.png
  reports/training/gradnorm.png
  reports/training/entropy.png
  reports/training/eval_summary.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--state",
        type=Path,
        default=ROOT / "runs/qlora-qwen2.5-1.5b-v1/checkpoint-174/trainer_state.json",
    )
    ap.add_argument("--out", type=Path, default=ROOT / "reports/training")
    args = ap.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)

    train_logs = [e for e in state["log_history"] if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in state["log_history"] if "eval_loss" in e]

    steps = [e["step"] for e in train_logs]
    losses = [e["loss"] for e in train_logs]
    accs = [e["mean_token_accuracy"] for e in train_logs]
    lrs = [e["learning_rate"] for e in train_logs]
    grads = [e["grad_norm"] for e in train_logs]
    ents = [e["entropy"] for e in train_logs]

    eval_steps = [e["step"] for e in eval_logs]
    eval_losses = [e["eval_loss"] for e in eval_logs]
    eval_accs = [e["eval_mean_token_accuracy"] for e in eval_logs]
    eval_ents = [e.get("eval_entropy") for e in eval_logs]

    plt.style.use("seaborn-v0_8-whitegrid")

    # Loss curve (train + eval overlay)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, losses, label="Train loss", color="#1f77b4", linewidth=2)
    ax.scatter(eval_steps, eval_losses, color="#d62728", s=80, zorder=5,
               label="Eval loss", edgecolors="black", linewidths=0.7)
    for s, v in zip(eval_steps, eval_losses):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(8, 8), fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Loss")
    ax.set_title("QLoRA Qwen2.5-1.5B — Train & Eval Loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "loss_curve.png", dpi=120)
    plt.close(fig)

    # Accuracy curve
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, accs, label="Train mean-token accuracy", color="#2ca02c", linewidth=2)
    ax.scatter(eval_steps, eval_accs, color="#d62728", s=80, zorder=5,
               label="Eval mean-token accuracy", edgecolors="black", linewidths=0.7)
    for s, v in zip(eval_steps, eval_accs):
        ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                    xytext=(8, -12), fontsize=9)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Mean token accuracy")
    ax.set_title("QLoRA Qwen2.5-1.5B — Token Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "accuracy_curve.png", dpi=120)
    plt.close(fig)

    # Learning rate schedule
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, lrs, color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    ax.set_title("Learning Rate Schedule (Cosine, 6% warmup, peak 3e-5)")
    fig.tight_layout()
    fig.savefig(args.out / "lr_schedule.png", dpi=120)
    plt.close(fig)

    # Grad norm
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, grads, color="#9467bd", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm")
    ax.set_title("Gradient Norm")
    fig.tight_layout()
    fig.savefig(args.out / "gradnorm.png", dpi=120)
    plt.close(fig)

    # Entropy
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, ents, label="Train entropy", color="#17becf", linewidth=2)
    ax.scatter(eval_steps, eval_ents, color="#d62728", s=80, zorder=5,
               label="Eval entropy", edgecolors="black", linewidths=0.7)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Entropy (nats)")
    ax.set_title("Token Distribution Entropy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "entropy.png", dpi=120)
    plt.close(fig)

    # Eval summary bar
    fig, ax = plt.subplots(figsize=(8, 4))
    epochs = [e["epoch"] for e in eval_logs]
    width = 0.35
    x = list(range(len(epochs)))
    ax.bar([i - width / 2 for i in x], eval_losses, width=width,
           label="Eval loss", color="#d62728", alpha=0.85)
    ax2 = ax.twinx()
    ax2.bar([i + width / 2 for i in x], eval_accs, width=width,
            label="Eval accuracy", color="#2ca02c", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"epoch {e:.0f}" for e in epochs])
    ax.set_ylabel("Eval loss", color="#d62728")
    ax2.set_ylabel("Eval token accuracy", color="#2ca02c")
    ax.set_title("Per-Epoch Eval Snapshot")
    fig.tight_layout()
    fig.savefig(args.out / "eval_summary.png", dpi=120)
    plt.close(fig)

    print(f"Wrote {len(list(args.out.glob('*.png')))} plots to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
