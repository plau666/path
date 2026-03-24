#!/usr/bin/env python
"""Parse training log and plot metrics using scienceplots.

Usage:
    python plot_training_logs.py --log output/mimic_dp_eps10/log_rank0.txt
"""

import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import scienceplots  # noqa: F401


def parse_log(log_path: str):
    """Extract all metrics from the training log file."""
    # Step-level training metrics
    train_steps, losses, epsilons, clipped_pcts = [], [], [], []
    grad_avg, grad_max, lrs = [], [], []

    # Eval metrics
    eval_steps, eval_losses, eval_accs = [], [], []

    # Grad diagnostics (logged every 10 steps)
    diag_steps = []
    preclip_min, preclip_mean, preclip_max = [], [], []
    snr_vals = []
    postnoise_norms = []

    # Regex patterns
    train_re = re.compile(
        r"Step (\d+)/\d+ \| Loss ([\d.]+) \| Eps ([\d.]+) \| "
        r"Clipped ([\d.]+)% \| GradNorm avg=([\d.]+) max=([\d.]+) \| "
        r"LR ([\d.e+-]+)"
    )
    eval_re = re.compile(
        r"Step (\d+) \| Eval Loss ([\d.]+) \| Eval Acc ([\d.]+)%"
    )
    preclip_re = re.compile(
        r"Pre-clip\s+per-sample norms: min=([\d.]+) mean=([\d.]+) max=([\d.]+)"
    )
    snr_re = re.compile(r"SNR \(signal/noise\): ([\d.]+)")
    postnoise_re = re.compile(r"Post-noise grad norm \(step (\d+)\): ([\d.]+)")

    with open(log_path) as f:
        for line in f:
            m = train_re.search(line)
            if m:
                train_steps.append(int(m.group(1)))
                losses.append(float(m.group(2)))
                epsilons.append(float(m.group(3)))
                clipped_pcts.append(float(m.group(4)))
                grad_avg.append(float(m.group(5)))
                grad_max.append(float(m.group(6)))
                lrs.append(float(m.group(7)))
                continue

            m = eval_re.search(line)
            if m:
                eval_steps.append(int(m.group(1)))
                eval_losses.append(float(m.group(2)))
                eval_accs.append(float(m.group(3)))
                continue

            m = preclip_re.search(line)
            if m:
                preclip_min.append(float(m.group(1)))
                preclip_mean.append(float(m.group(2)))
                preclip_max.append(float(m.group(3)))
                continue

            m = snr_re.search(line)
            if m:
                snr_vals.append(float(m.group(1)))
                continue

            m = postnoise_re.search(line)
            if m:
                diag_steps.append(int(m.group(1)))
                postnoise_norms.append(float(m.group(2)))

    return {
        "train_steps": train_steps, "losses": losses, "epsilons": epsilons,
        "clipped_pcts": clipped_pcts, "grad_avg": grad_avg, "grad_max": grad_max,
        "lrs": lrs,
        "eval_steps": eval_steps, "eval_losses": eval_losses, "eval_accs": eval_accs,
        "diag_steps": diag_steps, "preclip_min": preclip_min,
        "preclip_mean": preclip_mean, "preclip_max": preclip_max,
        "snr_vals": snr_vals, "postnoise_norms": postnoise_norms,
    }


def plot_metrics(data, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.style.use(["science", "no-latex"])

    # 1. Training Loss
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["train_steps"], data["losses"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss")
    fig.savefig(output_dir / "training_loss.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 2. Eval Loss & Accuracy (twin y-axis)
    if data["eval_steps"]:
        fig, ax1 = plt.subplots(figsize=(6, 4))
        color1, color2 = "#1f77b4", "#d62728"
        ax1.plot(data["eval_steps"], data["eval_losses"], "o-", color=color1, label="Eval Loss")
        ax1.set_xlabel("Step")
        ax1.set_ylabel("Eval Loss", color=color1)
        ax1.tick_params(axis="y", labelcolor=color1)

        ax2 = ax1.twinx()
        ax2.plot(data["eval_steps"], data["eval_accs"], "s-", color=color2, label="Eval Acc")
        ax2.set_ylabel("Eval Accuracy (%)", color=color2)
        ax2.tick_params(axis="y", labelcolor=color2)

        ax1.set_title("Eval Loss & Accuracy")
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
        fig.savefig(output_dir / "eval_metrics.png", dpi=200, bbox_inches="tight")
        plt.close(fig)

    # 3. Privacy Budget (Epsilon)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["train_steps"], data["epsilons"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Epsilon")
    ax.set_title("Privacy Budget (Epsilon)")
    fig.savefig(output_dir / "epsilon.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 4. Pre-clip Gradient Norms
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.fill_between(data["diag_steps"], data["preclip_min"], data["preclip_max"],
                    alpha=0.2, label="min-max range")
    ax.plot(data["diag_steps"], data["preclip_mean"], label="mean")
    ax.set_xlabel("Step")
    ax.set_ylabel("Per-sample Gradient Norm")
    ax.set_title("Pre-clip Per-sample Gradient Norms")
    ax.legend()
    fig.savefig(output_dir / "preclip_norms.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 5. SNR (Signal-to-Noise Ratio)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["diag_steps"], data["snr_vals"])
    ax.set_xlabel("Step")
    ax.set_ylabel("SNR")
    ax.set_title("Signal-to-Noise Ratio")
    fig.savefig(output_dir / "snr.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # 6. Learning Rate
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(data["train_steps"], data["lrs"])
    ax.set_xlabel("Step")
    ax.set_ylabel("Learning Rate")
    ax.set_title("Learning Rate Schedule")
    fig.savefig(output_dir / "learning_rate.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved 6 plots to {output_dir}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=str, required=True, help="Path to log_rank0.txt")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Directory for plots (default: same dir as log)")
    args = parser.parse_args()

    log_path = Path(args.log)
    output_dir = Path(args.output_dir) if args.output_dir else log_path.parent / "plots"

    data = parse_log(str(log_path))
    print(f"Parsed: {len(data['train_steps'])} train steps, "
          f"{len(data['eval_steps'])} eval steps, "
          f"{len(data['diag_steps'])} grad diag steps")
    plot_metrics(data, output_dir)


if __name__ == "__main__":
    main()
