import json
import os
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


def suggest_common_params(trial, use_temp_weight: bool = False) -> Dict[str, Any]:
    """Sample a compact but effective shared search space."""
    params: Dict[str, Any] = {
        "learning_rate": trial.suggest_float("learning_rate", 1e-4, 5e-3, log=True),
        "gamma": trial.suggest_float("gamma", 0.90, 0.99),
        "co2_weight": trial.suggest_float("co2_weight", 0.2, 0.8),
    }
    weight_name = "temp_weight" if use_temp_weight else "pmv_weight"
    params[weight_name] = trial.suggest_float(weight_name, 0.2, 0.8)
    return params


def objective_score(
    episode_metrics: Dict[str, Any],
    *,
    power_coef: float,
    pmv_coef: float,
    co2_coef: float,
) -> float:
    """Single scalar objective: maximize reward while penalizing energy/violations."""
    val_reward = float(episode_metrics.get("val_reward", 0.0))
    val_power = float(episode_metrics.get("val_power_kwh") or 0.0)
    val_pmv = float(episode_metrics.get("val_pmv_violation_pct") or 0.0)
    val_co2 = float(episode_metrics.get("val_co2_violation_pct") or 0.0)
    return val_reward - power_coef * val_power - pmv_coef * val_pmv - co2_coef * val_co2


def read_metrics_last_row(metrics_path: str) -> Dict[str, Any]:
    if not os.path.exists(metrics_path):
        raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
    with open(metrics_path, "r") as f:
        rows = json.load(f)
    if not rows:
        raise ValueError(f"metrics.json is empty: {metrics_path}")
    return rows[-1]


def _latest_subdir(path: str) -> str:
    root = Path(path)
    subdirs = [p for p in root.iterdir() if p.is_dir()]
    if not subdirs:
        raise FileNotFoundError(f"No run subdirectory found under: {path}")
    latest = max(subdirs, key=lambda p: p.stat().st_mtime)
    return str(latest)


def run_training_trial(
    command: List[str],
    *,
    output_root: str,
    timeout_seconds: Optional[int] = None,
    env: Optional[Dict[str, str]] = None,
) -> Dict[str, str]:
    """
    Execute one training trial and return run artifact paths.
    """
    os.makedirs(output_root, exist_ok=True)
    start = time.time()
    subprocess.run(
        command,
        check=True,
        timeout=timeout_seconds,
        env=env,
    )
    elapsed = time.time() - start

    run_dir = _latest_subdir(output_root)
    metrics_path = os.path.join(run_dir, "metrics.json")
    summary_path = os.path.join(run_dir, "training_summary.json")
    return {
        "run_dir": run_dir,
        "metrics_path": metrics_path,
        "summary_path": summary_path,
        "elapsed_seconds": f"{elapsed:.2f}",
    }


def build_trial_output_root(base_dir: str, prefix: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(base_dir, f"{prefix}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path
