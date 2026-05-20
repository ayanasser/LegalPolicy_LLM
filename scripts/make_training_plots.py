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
    ap.add_argument("--title-suffix", type=str, default="",
                    help="Appended to chart titles, e.g. 'Qwen2.5-3B + Unsloth knowledge'.")
    args = ap.parse_args()

    state = json.loads(args.state.read_text(encoding="utf-8"))
    args.out.mkdir(parents=True, exist_ok=True)
    suffix = f" — {args.title_suffix}" if args.title_suffix else ""

    train_logs = [e for e in state["log_history"] if "loss" in e and "eval_loss" not in e]
    eval_logs = [e for e in state["log_history"] if "eval_loss" in e]

    # Some trainer logs (e.g. Unsloth) don't include mean_token_accuracy / entropy
    # in the *train* stream — only at eval. Plot what we have, skip what we don't.
    has_train_acc = bool(train_logs) and "mean_token_accuracy" in train_logs[0]
    has_train_ent = bool(train_logs) and "entropy" in train_logs[0]
    has_eval_acc = bool(eval_logs) and "eval_mean_token_accuracy" in eval_logs[0]
    has_eval_ent = bool(eval_logs) and "eval_entropy" in eval_logs[0]

    steps = [e["step"] for e in train_logs]
    losses = [e["loss"] for e in train_logs]
    accs = [e["mean_token_accuracy"] for e in train_logs] if has_train_acc else []
    lrs = [e["learning_rate"] for e in train_logs]
    grads = [e["grad_norm"] for e in train_logs]
    ents = [e["entropy"] for e in train_logs] if has_train_ent else []

    eval_steps = [e["step"] for e in eval_logs]
    eval_losses = [e["eval_loss"] for e in eval_logs]
    eval_accs = [e["eval_mean_token_accuracy"] for e in eval_logs] if has_eval_acc else []
    eval_ents = [e.get("eval_entropy") for e in eval_logs] if has_eval_ent else []

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
    ax.set_title(f"Train & Eval Loss{suffix}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.out / "loss_curve.png", dpi=120)
    plt.close(fig)

    # Accuracy curve (only if logged)
    if has_train_acc or has_eval_acc:
        fig, ax = plt.subplots(figsize=(9, 5))
        if has_train_acc:
            ax.plot(steps, accs, label="Train mean-token accuracy", color="#2ca02c", linewidth=2)
        if has_eval_acc:
            ax.scatter(eval_steps, eval_accs, color="#d62728", s=80, zorder=5,
                       label="Eval mean-token accuracy", edgecolors="black", linewidths=0.7)
            for s, v in zip(eval_steps, eval_accs):
                ax.annotate(f"{v:.3f}", (s, v), textcoords="offset points",
                            xytext=(8, -12), fontsize=9)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Mean token accuracy")
        ax.set_title(f"Token Accuracy{suffix}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "accuracy_curve.png", dpi=120)
        plt.close(fig)

    # Learning rate schedule
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, lrs, color="#ff7f0e", linewidth=2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Learning rate")
    peak_lr = max(lrs) if lrs else 0
    ax.set_title(f"Learning Rate Schedule (peak {peak_lr:.1e}){suffix}")
    fig.tight_layout()
    fig.savefig(args.out / "lr_schedule.png", dpi=120)
    plt.close(fig)

    # Grad norm
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(steps, grads, color="#9467bd", linewidth=1.5)
    ax.set_xlabel("Training step")
    ax.set_ylabel("Gradient norm")
    ax.set_title(f"Gradient Norm{suffix}")
    fig.tight_layout()
    fig.savefig(args.out / "gradnorm.png", dpi=120)
    plt.close(fig)

    # Entropy (only if logged)
    if has_train_ent or has_eval_ent:
        fig, ax = plt.subplots(figsize=(9, 5))
        if has_train_ent:
            ax.plot(steps, ents, label="Train entropy", color="#17becf", linewidth=2)
        if has_eval_ent:
            ax.scatter(eval_steps, eval_ents, color="#d62728", s=80, zorder=5,
                       label="Eval entropy", edgecolors="black", linewidths=0.7)
        ax.set_xlabel("Training step")
        ax.set_ylabel("Entropy (nats)")
        ax.set_title(f"Token Distribution Entropy{suffix}")
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.out / "entropy.png", dpi=120)
        plt.close(fig)

    # Eval summary bar (loss always; accuracy if logged)
    if eval_logs:
        fig, ax = plt.subplots(figsize=(8, 4))
        epochs = [e["epoch"] for e in eval_logs]
        x = list(range(len(epochs)))
        if has_eval_acc:
            width = 0.35
            ax.bar([i - width / 2 for i in x], eval_losses, width=width,
                   label="Eval loss", color="#d62728", alpha=0.85)
            ax2 = ax.twinx()
            ax2.bar([i + width / 2 for i in x], eval_accs, width=width,
                    label="Eval accuracy", color="#2ca02c", alpha=0.85)
            ax2.set_ylabel("Eval token accuracy", color="#2ca02c")
        else:
            ax.bar(x, eval_losses, color="#d62728", alpha=0.85, label="Eval loss")
            for i, v in zip(x, eval_losses):
                ax.text(i, v, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        ax.set_xticks(x)
        ax.set_xticklabels([f"epoch {e:.0f}" for e in epochs])
        ax.set_ylabel("Eval loss", color="#d62728")
        ax.set_title(f"Per-Epoch Eval Snapshot{suffix}")
        fig.tight_layout()
        fig.savefig(args.out / "eval_summary.png", dpi=120)
        plt.close(fig)

    print(f"Wrote {len(list(args.out.glob('*.png')))} plots to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
