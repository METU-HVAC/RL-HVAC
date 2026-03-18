import json
import os
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np


def _load_metrics(metrics_path: str) -> List[dict]:
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    if not isinstance(metrics, list):
        raise ValueError(f"Expected a list in metrics file: {metrics_path}")
    if not metrics:
        raise ValueError(f"No episode metrics found in: {metrics_path}")
    return metrics


def _series(metrics: List[dict], key: str) -> List[float]:
    values = []
    for row in metrics:
        value = row.get(key)
        values.append(np.nan if value is None else float(value))
    return values


def plot_training_curves(metrics_path: str, output_dir: str, run_label: Optional[str] = None) -> str:
    metrics = _load_metrics(metrics_path)
    episodes = [int(row["episode"]) for row in metrics]

    train_reward = _series(metrics, "train_reward")
    val_reward = _series(metrics, "val_reward")
    train_loss = _series(metrics, "train_loss")
    val_power = _series(metrics, "val_power_kwh")
    val_pmv = _series(metrics, "val_pmv_violation_pct")
    val_co2 = _series(metrics, "val_co2_violation_pct")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    title = f"Training Core Metrics - {run_label}" if run_label else "Training Core Metrics"
    fig.suptitle(title, fontsize=14, fontweight="bold")

    ax = axes[0, 0]
    ax.plot(episodes, train_reward, marker="o", label="Train Reward")
    ax.plot(episodes, val_reward, marker="o", label="Val Reward")
    ax.set_title("Reward")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(episodes, train_loss, marker="o", color="#d35400")
    ax.set_title("Train Loss")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(episodes, val_power, marker="o", color="#2c3e50")
    ax.set_title("Validation Energy (kWh)")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(episodes, val_pmv, marker="o", color="#8e44ad")
    ax.set_title("Validation PMV Violation (%)")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    ax = axes[2, 0]
    ax.plot(episodes, val_co2, marker="o", color="#16a085")
    ax.set_title("Validation CO2 Violation (%)")
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    # Reserve one panel for algorithm-specific extras when present.
    ax = axes[2, 1]
    has_split_rewards = any("val_reward_fan" in row for row in metrics)
    if has_split_rewards:
        fan_reward = _series(metrics, "val_reward_fan")
        hvac_reward = _series(metrics, "val_reward_hvac")
        ax.plot(episodes, fan_reward, marker="o", label="Val Fan Reward")
        ax.plot(episodes, hvac_reward, marker="o", label="Val HVAC Reward")
        ax.legend()
        ax.set_title("MADQN Split Rewards")
    else:
        ax.axis("off")
        ax.text(0.5, 0.5, "Reserved", ha="center", va="center", alpha=0.4)
    ax.set_xlabel("Episode")
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "training_core_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_comparison_curves(metrics_paths: List[str], labels: List[str], output_dir: str) -> str:
    if len(metrics_paths) != len(labels):
        raise ValueError("metrics_paths and labels must have the same length")
    if not metrics_paths:
        raise ValueError("At least one metrics file must be provided")

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle("Training Core Metrics Comparison", fontsize=14, fontweight="bold")

    metric_specs = [
        ("train_reward", "Train Reward"),
        ("val_reward", "Val Reward"),
        ("train_loss", "Train Loss"),
        ("val_power_kwh", "Validation Energy (kWh)"),
        ("val_pmv_violation_pct", "Validation PMV Violation (%)"),
        ("val_co2_violation_pct", "Validation CO2 Violation (%)"),
    ]

    for metrics_path, label in zip(metrics_paths, labels):
        metrics = _load_metrics(metrics_path)
        episodes = [int(row["episode"]) for row in metrics]
        for idx, (key, title) in enumerate(metric_specs):
            r, c = divmod(idx, 2)
            values = _series(metrics, key)
            axes[r, c].plot(episodes, values, marker="o", label=label)
            axes[r, c].set_title(title)
            axes[r, c].set_xlabel("Episode")
            axes[r, c].grid(alpha=0.3)

    for r in range(3):
        for c in range(2):
            axes[r, c].legend()

    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_core_metrics.png")
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output_path
