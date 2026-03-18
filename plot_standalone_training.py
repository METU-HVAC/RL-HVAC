#!/usr/bin/env python3
"""
Offline plotting helper for standalone training runs.

Examples:
    python plot_standalone_training.py --run-dir ./standalone_results/xxx
    python plot_standalone_training.py --run-dir ./run1 --run-dir ./run2 --label dqn --label madqn
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.standalone_plots import plot_comparison_curves, plot_training_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Plot standalone training metrics.")
    parser.add_argument(
        "--run-dir",
        action="append",
        required=True,
        help="Path to standalone run directory containing metrics.json. Repeat for multiple runs.",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=None,
        help="Optional label for each --run-dir (same order).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output directory. Default: first run dir.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dirs = args.run_dir
    labels = args.label or [os.path.basename(os.path.normpath(rd)) for rd in run_dirs]
    if len(labels) != len(run_dirs):
        raise ValueError("Number of --label entries must match number of --run-dir entries.")

    metrics_paths = [os.path.join(rd, "metrics.json") for rd in run_dirs]
    for metrics_path in metrics_paths:
        if not os.path.exists(metrics_path):
            raise FileNotFoundError(f"metrics.json not found: {metrics_path}")

    output_dir = args.output_dir or run_dirs[0]
    os.makedirs(output_dir, exist_ok=True)

    if len(metrics_paths) == 1:
        out = plot_training_curves(
            metrics_path=metrics_paths[0],
            output_dir=output_dir,
            run_label=labels[0],
        )
        print(f"Saved: {out}")
    else:
        out = plot_comparison_curves(metrics_paths=metrics_paths, labels=labels, output_dir=output_dir)
        print(f"Saved: {out}")


if __name__ == "__main__":
    main()
