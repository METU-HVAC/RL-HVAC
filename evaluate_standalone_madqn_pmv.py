#!/usr/bin/env python3
"""
Standalone evaluation script for MADQN PMV + CO2 HVAC/Fan control.
Runs WITHOUT WandB and supports optional semantic augmentation.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sinergym  # noqa: F401

from algorithms.dqn.dqn import DQNAgent, device
from environments.environment import create_environment
from environments.reward import CO2andPMVReward
from experiments.madqn.madqn_train_fully_cooperative import (
    all_action_map,
    combine_actions,
    fan_map,
    get_agent_observation_dict_based,
    hvac_map,
    observation_variables,
)
from utils.dataset import balanced_month_sample, generate_chunks
from utils.experiment_utils import (
    add_observation,
    append_fan_speed_to_dict,
    append_info_and_time_to_dict_pmv,
    append_raw_action_to_dict,
    min_max_normalize,
    obs_maxs_pmv,
    obs_mins_pmv,
)
from utils.standalone_common import (
    add_common_eval_args,
    add_semantic_args,
    build_eval_run_name,
    build_semantic_provider,
    semantic_args_metadata,
)


def get_semantic_augmented_observation(
    base_obs: List[float],
    raw_state,
    semantic_provider,
    mode: str = "concat",
) -> List[float]:
    if isinstance(raw_state, torch.Tensor):
        raw_state = raw_state.squeeze().tolist()
    elif isinstance(raw_state, np.ndarray):
        raw_state = raw_state.tolist()

    snapshot = dict(zip(observation_variables, raw_state))
    z_t = semantic_provider.get_state_from_snapshot(snapshot).tolist()
    if mode == "latent":
        return z_t
    return base_obs + z_t


def run_evaluation(
    env_id: str,
    start_date,
    end_date,
    season: str,
    steps_per_chunk: int,
    fan_agent: DQNAgent,
    hvac_agent: DQNAgent,
    timesteps_per_hour: int,
    reward_config: dict,
    semantic_provider=None,
    semantic_mode: str = "concat",
):
    env = create_environment(
        env_id,
        start_date,
        end_date,
        season,
        CO2andPMVReward,
        episode_type="Validation",
        timesteps_per_hour=timesteps_per_hour,
        reward_kwargs=reward_config,
    )

    state, info = env.reset()
    combined_action = 20
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    current_step = 1
    total_fan_reward = 0.0
    total_hvac_reward = 0.0
    all_obs_dict: Dict[str, List[float]] = {}

    while current_step < steps_per_chunk:
        normalized_state = torch.tensor(
            min_max_normalize(state, obs_mins_pmv, obs_maxs_pmv),
            dtype=torch.float32,
            device=device,
        )
        fan_obs = get_agent_observation_dict_based("WindowFan", normalized_state, action=combined_action)
        hvac_obs = get_agent_observation_dict_based("HVAC", normalized_state, action=combined_action)
        if semantic_provider is not None:
            fan_obs = get_semantic_augmented_observation(fan_obs, state, semantic_provider, semantic_mode)
            hvac_obs = get_semantic_augmented_observation(hvac_obs, state, semantic_provider, semantic_mode)

        fan_obs_tensor = torch.tensor(fan_obs, dtype=torch.float32, device=device).unsqueeze(0)
        hvac_obs_tensor = torch.tensor(hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)

        fan_action = fan_agent.choose_greedy_action(fan_obs_tensor)
        hvac_action = hvac_agent.choose_greedy_action(hvac_obs_tensor)
        combined_action = combine_actions(
            fan_action.item(),
            hvac_action.item(),
            all_action_map,
            fan_map,
            hvac_map,
        )

        next_state, _, terminated, truncated, info = env.step(combined_action)
        done = terminated or truncated

        total_fan_reward += info["co2_term"] + info["window_energy_term"]
        total_hvac_reward += info["pmv_term"] + info["ac_energy_term"]

        state = next_state
        obs_dict = dict(zip(env.get_wrapper_attr("observation_variables"), state))
        obs_dict = append_info_and_time_to_dict_pmv(obs_dict, info, current_step, timesteps_per_hour)
        obs_dict = append_fan_speed_to_dict(
            obs_dict, all_action_map[combined_action][3], all_action_map[combined_action][2]
        )
        obs_dict = append_raw_action_to_dict(obs_dict, combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)

        if done:
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        current_step += 1

    env.close()
    return total_fan_reward, total_hvac_reward, all_obs_dict


def cleanup_simulation_folders():
    eplus_folders = glob.glob("Eplus-env-*")
    if eplus_folders:
        print(f"\nCleaning up {len(eplus_folders)} simulation folders...")
        for folder in eplus_folders:
            try:
                shutil.rmtree(folder)
            except Exception as exc:
                print(f"Warning: Could not remove {folder}: {exc}")


def generate_plots(obs_dict, output_dir, season):
    """Generate evaluation plots for MADQN PMV."""
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8-darkgrid")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f"MADQN PMV Evaluation - {season.upper()} Season", fontsize=14, fontweight="bold")

    timesteps = range(len(obs_dict.get("air_temperatures", [])))

    # 1) Temperature
    ax1 = axes[0, 0]
    if "air_temperatures" in obs_dict and "outdoor_temperatures" in obs_dict:
        ax1.plot(timesteps, obs_dict["air_temperatures"], label="Indoor", color="#e74c3c", linewidth=1)
        ax1.plot(
            timesteps,
            obs_dict["outdoor_temperatures"],
            label="Outdoor",
            color="#3498db",
            linewidth=1,
            alpha=0.7,
        )
        ax1.axhline(y=23, color="green", linestyle="--", alpha=0.5, label="Comfort Range")
        ax1.axhline(y=26, color="green", linestyle="--", alpha=0.5)
        ax1.fill_between(timesteps, 23, 26, alpha=0.1, color="green")
        ax1.set_ylabel("Temperature (C)")
        ax1.set_title("Indoor vs Outdoor Temperature")
        ax1.legend(loc="upper right", fontsize=8)

    # 2) PMV
    ax2 = axes[0, 1]
    if "pmvs" in obs_dict:
        pmvs = np.array(obs_dict["pmvs"])
        occupants = np.array(obs_dict.get("people_occupants", [1] * len(pmvs)))
        pmv_masked = np.where(occupants > 0, pmvs, np.nan)
        ax2.plot(timesteps, pmv_masked, color="#9b59b6", linewidth=1)
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Comfort Bounds (±0.5)")
        ax2.axhline(y=-0.5, color="red", linestyle="--", alpha=0.5)
        ax2.fill_between(timesteps, -0.5, 0.5, alpha=0.1, color="green")
        ax2.set_ylabel("PMV Index")
        ax2.set_title("Thermal Comfort (PMV)")
        ax2.set_ylim(-3, 3)
        ax2.legend(loc="upper right", fontsize=8)

    # 3) Power
    ax3 = axes[1, 0]
    if "total_electricity_HVACs" in obs_dict:
        hvac_power = np.array(obs_dict["total_electricity_HVACs"]) / 1000
        fan_power = np.array(obs_dict.get("window_fan_energies", [0] * len(hvac_power))) / 1000
        ax3.fill_between(timesteps, 0, hvac_power, alpha=0.7, label="HVAC", color="#e74c3c")
        ax3.fill_between(
            timesteps,
            hvac_power,
            hvac_power + fan_power,
            alpha=0.7,
            label="Fan",
            color="#3498db",
        )
        ax3.set_ylabel("Power (kW)")
        ax3.set_title("Energy Consumption")
        ax3.legend(loc="upper right", fontsize=8)

    # 4) CO2
    ax4 = axes[1, 1]
    if "air_co2s" in obs_dict:
        co2 = obs_dict["air_co2s"]
        ax4.plot(timesteps, co2, color="#27ae60", linewidth=1)
        ax4.axhline(y=800, color="orange", linestyle="--", alpha=0.7, label="Threshold (800 ppm)")
        ax4.axhline(y=1000, color="red", linestyle="--", alpha=0.5, label="High (1000 ppm)")
        ax4.set_ylabel("CO2 (ppm)")
        ax4.set_title("Air Quality (CO2)")
        ax4.legend(loc="upper right", fontsize=8)

    # 5) Combined actions
    ax5 = axes[2, 0]
    if "raw_actions" in obs_dict:
        actions = obs_dict["raw_actions"]
        ax5.scatter(timesteps, actions, s=1, alpha=0.5, c="#8e44ad")
        ax5.set_ylabel("Action Index")
        ax5.set_title("Combined Actions Over Time")
        ax5.set_ylim(-1, 41)
        ax5.set_xlabel("Timestep")

    # 6) MADQN-specific split controls
    ax6 = axes[2, 1]
    if "window_fan_speeds" in obs_dict and "ac_fan_speeds" in obs_dict:
        ax6.plot(timesteps, obs_dict["window_fan_speeds"], label="Window Fan Speed", color="#f39c12", linewidth=1)
        ax6.plot(timesteps, obs_dict["ac_fan_speeds"], label="AC Fan Flag", color="#2ecc71", linewidth=1)
        ax6.set_ylabel("Control Signal")
        ax6.set_title("MADQN Agent Control Traces")
        ax6.set_xlabel("Timestep")
        ax6.legend(loc="upper right", fontsize=8)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "evaluation_plots.png"), dpi=150, bbox_inches="tight")
    plt.close()

    create_summary_plot(obs_dict, output_dir, season)


def create_summary_plot(obs_dict, output_dir, season):
    """Create a compact summary plot for PMV runs."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))

    if "air_temperatures" in obs_dict:
        temps = np.array(obs_dict["air_temperatures"])
        hours = len(temps) // 4 if len(temps) > 4 else 1
        hourly_temps = [np.mean(temps[i * 4 : (i + 1) * 4]) for i in range(hours)]
        ax.plot(range(hours), hourly_temps, label="Indoor Temp", color="#e74c3c", linewidth=2)
        ax.axhspan(23, 26, alpha=0.2, color="green", label="Comfort Zone")
        ax.set_xlabel("Hours")
        ax.set_ylabel("Temperature (C)")
        ax.set_title(f"Hourly Temperature Profile - {season.upper()} Season")
        ax.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "temperature_profile.png"), dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Standalone MADQN PMV Evaluation (No WandB)")
    parser.add_argument("--model-path-fan", type=str, required=True, help="Path to fan model (.pth)")
    parser.add_argument("--model-path-hvac", type=str, required=True, help="Path to HVAC model (.pth)")
    add_common_eval_args(parser, season_choices=["hot", "cool", "mixed", "ankara"])
    parser.add_argument("--co2-weight", type=float, default=0.5)
    parser.add_argument("--pmv-weight", type=float, default=0.4)
    parser.add_argument("--gamma", type=float, default=0.95)
    add_semantic_args(parser)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    semantic_provider = None
    semantic_mode = None
    semantic_model_path = None
    semantic_latent_dim = 32
    if args.use_semantic:
        semantic_provider, semantic_mode, semantic_model_path = build_semantic_provider(
            args,
            device=str(device),
            semantic_root="./semantic",
        )
        semantic_latent_dim = semantic_provider.latent_dim
        print(f"Semantic enabled (mode={semantic_mode}, checkpoint={semantic_model_path})")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_eval_run_name(
        algo="madqn_pmv",
        env_id=args.env_id,
        season=args.season,
        model_ids=[Path(args.model_path_fan).stem, Path(args.model_path_hvac).stem],
        use_semantic=args.use_semantic,
        semantic_mode=args.semantic_mode,
        semantic_model_path=semantic_model_path,
        semantic_latent_dim=semantic_latent_dim if args.use_semantic else None,
        run_tag=args.run_tag,
        timestamp=timestamp,
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    fan_state_size = 5 if not args.use_semantic else (semantic_latent_dim if semantic_mode == "latent" else 5 + semantic_latent_dim)
    hvac_state_size = 9 if not args.use_semantic else (semantic_latent_dim if semantic_mode == "latent" else 9 + semantic_latent_dim)
    fan_action_size = 4
    hvac_action_size = 10
    timesteps_per_hour = 4
    days_per_chunk = 8
    steps_per_chunk = timesteps_per_hour * 24 * days_per_chunk

    reward_config = {
        "pmv_variables": ["pmv"],
        "co2_variable": "air_co2",
        "energy_variables": ["total_electricity_HVAC", "window_fan_energy"],
        "ac_energy_weight": 1 - args.pmv_weight,
        "fan_energy_weight": 1 - args.co2_weight,
        "co2_weight": args.co2_weight,
        "pmv_weight": args.pmv_weight,
        "lambda_energy": 1 / 1_600_000,
        "lambda_pmv": 1.0,
        "lambda_co2": 1.0,
        "co2_threshold": 800,
    }

    training_config = {
        "batch_size": 64,
        "gamma": args.gamma,
        "eps_start": 0.9,
        "eps_end": 0.01,
        "eps_decay": 5,
        "tau": 0.005,
        "lr": 3e-3,
        "memory_capacity": 2 * 52600,
        "layer_sizes": [256, 256, 256],
    }

    fan_agent = DQNAgent(fan_state_size, fan_action_size, 1, 1, training_config)
    hvac_agent = DQNAgent(hvac_state_size, hvac_action_size, 1, 1, training_config)
    fan_agent.load_model(args.model_path_fan)
    hvac_agent.load_model(args.model_path_hvac)

    start_date = datetime(1997, 1, 1)
    chunks = generate_chunks(start_date, days_per_chunk, 365, step_size=days_per_chunk, seasons=[args.season])
    _, val_chunks, _ = balanced_month_sample(chunks, val_chunks_per_month=1, seed=args.seed)

    fan_rewards = []
    hvac_rewards = []
    total_power_kwh = []
    pmv_violations = []
    co2_violations = []
    last_obs_dict = {}

    with tqdm(val_chunks, desc="Evaluating", ncols=100) as pbar:
        for chunk in pbar:
            fan_r, hvac_r, obs_dict = run_evaluation(
                args.env_id,
                *chunk,
                steps_per_chunk,
                fan_agent,
                hvac_agent,
                timesteps_per_hour,
                reward_config,
                semantic_provider=semantic_provider,
                semantic_mode=semantic_mode or "concat",
            )
            fan_rewards.append(fan_r)
            hvac_rewards.append(hvac_r)
            last_obs_dict = obs_dict

            if "total_electricity_HVACs" in obs_dict:
                hvac_power = sum(obs_dict["total_electricity_HVACs"])
                window_power = sum(obs_dict.get("window_fan_energies", [0]))
                total_power_kwh.append((hvac_power + window_power) / 3600000)

            if "pmv_violations" in obs_dict:
                vals = [v for v in obs_dict["pmv_violations"] if v is not None]
                if vals:
                    pmv_violations.append(sum(vals) / len(vals) * 100)

            if "co2_violations" in obs_dict:
                vals = [v for v in obs_dict["co2_violations"] if v is not None]
                if vals:
                    co2_violations.append(sum(vals) / len(vals) * 100)

            pbar.set_postfix({"FanR": f"{np.mean(fan_rewards):.1f}", "HVACR": f"{np.mean(hvac_rewards):.1f}"})

    results = {
        "env_id": args.env_id,
        "season": args.season,
        "model_path_fan": args.model_path_fan,
        "model_path_hvac": args.model_path_hvac,
        "avg_fan_reward": float(np.mean(fan_rewards)),
        "avg_hvac_reward": float(np.mean(hvac_rewards)),
        "avg_total_reward": float(np.mean(fan_rewards) + np.mean(hvac_rewards)),
        "avg_power_kwh": float(np.mean(total_power_kwh)) if total_power_kwh else None,
        "pmv_violation_pct": float(np.mean(pmv_violations)) if pmv_violations else None,
        "co2_violation_pct": float(np.mean(co2_violations)) if co2_violations else None,
        "semantic": semantic_args_metadata(args, fallback_model_path=semantic_model_path),
    }
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print("Evaluation complete")
    print(f"Fan reward: {results['avg_fan_reward']:.2f}")
    print(f"HVAC reward: {results['avg_hvac_reward']:.2f}")
    print(f"Results saved to: {results_path}")

    if last_obs_dict:
        trajectory_path = os.path.join(output_dir, "last_chunk_observations.json")
        with open(trajectory_path, "w") as f:
            json.dump(last_obs_dict, f, default=float)

    # Generate visualizations
    if last_obs_dict:
        print("\nGenerating visualizations...")
        generate_plots(last_obs_dict, output_dir, args.season)
        print(f"Plots saved to: {output_dir}")
    else:
        print("No observations available for plotting.")

    cleanup_simulation_folders()


if __name__ == "__main__":
    main()
