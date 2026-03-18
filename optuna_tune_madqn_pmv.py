#!/usr/bin/env python3
import argparse
import json
import os
from datetime import datetime

import optuna

from utils.optuna_tuning import (
    build_trial_output_root,
    objective_score,
    read_metrics_last_row,
    run_training_trial,
    suggest_common_params,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Optuna tuner for standalone MADQN-PMV.")
    parser.add_argument("--study-name", type=str, default="madqn_pmv_optuna")
    parser.add_argument("--study-dir", type=str, default="./optuna_runs")
    parser.add_argument("--n-trials", type=int, default=20)
    parser.add_argument("--season", type=str, default="hot", choices=["hot", "cool", "mixed", "ankara"])
    parser.add_argument("--env-id", type=str, default="A403mediumfanger")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=int, default=0)
    parser.add_argument("--power-coef", type=float, default=1.0)
    parser.add_argument("--pmv-coef", type=float, default=2.0)
    parser.add_argument("--co2-coef", type=float, default=2.0)
    parser.add_argument("--pruning", action="store_true")
    parser.add_argument("--use-semantic", action="store_true")
    parser.add_argument("--semantic-mode", type=str, default="concat", choices=["concat", "latent"])
    parser.add_argument("--semantic-model-path", type=str, default=None)
    parser.add_argument("--semantic-stats-path", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    semantic_requested = args.use_semantic or args.semantic_model_path is not None or args.semantic_stats_path is not None
    if semantic_requested and (not args.semantic_model_path or not args.semantic_stats_path):
        raise ValueError("When semantic is enabled, both --semantic-model-path and --semantic-stats-path are required.")

    os.makedirs(args.study_dir, exist_ok=True)
    study_root = build_trial_output_root(args.study_dir, args.study_name)
    storage = f"sqlite:///{os.path.join(study_root, 'optuna_study_madqn_pmv.db')}"

    pruner = optuna.pruners.MedianPruner(n_startup_trials=5) if args.pruning else optuna.pruners.NopPruner()
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=True,
        direction="maximize",
        pruner=pruner,
    )

    def objective(trial):
        params = suggest_common_params(trial, use_temp_weight=False)
        trial_root = os.path.join(study_root, "trials", f"trial_{trial.number:04d}")
        os.makedirs(trial_root, exist_ok=True)

        cmd = [
            "python",
            "train_standalone_madqn_pmv.py",
            "--episodes",
            str(args.episodes),
            "--season",
            args.season,
            "--env-id",
            args.env_id,
            "--seed",
            str(args.seed),
            "--output-dir",
            trial_root,
            "--learning-rate",
            str(params["learning_rate"]),
            "--gamma",
            str(params["gamma"]),
            "--co2-weight",
            str(params["co2_weight"]),
            "--pmv-weight",
            str(params["pmv_weight"]),
        ]
        if semantic_requested:
            cmd.extend(
                [
                    "--use-semantic",
                    "--semantic-mode",
                    args.semantic_mode,
                    "--semantic-model-path",
                    args.semantic_model_path,
                    "--semantic-stats-path",
                    args.semantic_stats_path,
                ]
            )

        run_meta = run_training_trial(
            cmd,
            output_root=trial_root,
            timeout_seconds=args.timeout_seconds if args.timeout_seconds > 0 else None,
            env=os.environ.copy(),
        )
        last_row = read_metrics_last_row(run_meta["metrics_path"])
        score = objective_score(
            last_row,
            power_coef=args.power_coef,
            pmv_coef=args.pmv_coef,
            co2_coef=args.co2_coef,
        )
        trial.report(score, step=0)
        if trial.should_prune():
            raise optuna.TrialPruned()
        return score

    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)

    best = {
        "study_name": args.study_name,
        "created_at": datetime.now().isoformat(),
        "best_value": study.best_value,
        "best_params": study.best_params,
    }
    best_path = os.path.join(study_root, "best_params_madqn_pmv.json")
    with open(best_path, "w") as f:
        json.dump(best, f, indent=2)

    trials_csv_path = os.path.join(study_root, "trials_madqn_pmv.csv")
    study.trials_dataframe().to_csv(trials_csv_path, index=False)

    print(f"Best value: {study.best_value:.4f}")
    print(f"Best params saved to: {best_path}")
    print(f"Trials saved to: {trials_csv_path}")
    print(f"Study DB: {storage}")


if __name__ == "__main__":
    main()
