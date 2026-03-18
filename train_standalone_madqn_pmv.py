#!/usr/bin/env python3
"""
Standalone training script for MADQN PMV + CO2 HVAC/Fan control.
Runs WITHOUT WandB and supports optional semantic augmentation.
"""

import argparse
import glob
import json
import os
import random
import shutil
import sys
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
    add_common_train_args,
    add_semantic_args,
    build_train_run_name,
    build_semantic_provider,
    semantic_args_metadata,
)
from utils.standalone_plots import plot_training_curves


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


def run_episode(
    env_id: str,
    start_date,
    end_date,
    season: str,
    episode_type: str,
    steps_per_chunk: int,
    fan_agent: DQNAgent,
    hvac_agent: DQNAgent,
    train_interval: int,
    timesteps_per_hour: int,
    reward_config: dict,
    switching_penalty: float = 0.0,
    semantic_provider=None,
    semantic_mode: str = "concat",
):
    env = create_environment(
        env_id,
        start_date,
        end_date,
        season,
        CO2andPMVReward,
        episode_type=episode_type,
        timesteps_per_hour=timesteps_per_hour,
        reward_kwargs=reward_config,
    )

    state, info = env.reset()
    combined_action = 20
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

    current_step = 1
    total_fan_reward = 0.0
    total_hvac_reward = 0.0
    fan_losses: List[float] = []
    hvac_losses: List[float] = []
    all_obs_dict: Dict[str, List[float]] = {}
    prev_fan_action = None
    prev_hvac_action = None

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

        if episode_type == "Training":
            fan_action = fan_agent.select_action(fan_obs_tensor)
            hvac_action = hvac_agent.select_action(hvac_obs_tensor)
        else:
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

        fan_reward_val = info["co2_term"] + info["window_energy_term"]
        hvac_reward_val = info["pmv_term"] + info["ac_energy_term"]

        # Optional switching penalties per agent.
        if prev_fan_action is not None and prev_fan_action != fan_action.item():
            fan_reward_val -= switching_penalty
        if prev_hvac_action is not None and prev_hvac_action != hvac_action.item():
            hvac_reward_val -= switching_penalty

        fan_reward = torch.tensor([fan_reward_val], dtype=torch.float32, device=device)
        hvac_reward = torch.tensor([hvac_reward_val], dtype=torch.float32, device=device)

        normalized_next_state = torch.tensor(
            min_max_normalize(next_state, obs_mins_pmv, obs_maxs_pmv),
            dtype=torch.float32,
            device=device,
        )
        next_fan_obs = get_agent_observation_dict_based(
            "WindowFan", normalized_next_state, action=combined_action
        )
        next_hvac_obs = get_agent_observation_dict_based(
            "HVAC", normalized_next_state, action=combined_action
        )
        if semantic_provider is not None:
            next_fan_obs = get_semantic_augmented_observation(
                next_fan_obs, next_state, semantic_provider, semantic_mode
            )
            next_hvac_obs = get_semantic_augmented_observation(
                next_hvac_obs, next_state, semantic_provider, semantic_mode
            )

        next_fan_obs_tensor = torch.tensor(next_fan_obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_hvac_obs_tensor = torch.tensor(next_hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)

        if episode_type == "Training":
            fan_agent.store_transition(fan_obs_tensor, fan_action, next_fan_obs_tensor, fan_reward)
            hvac_agent.store_transition(hvac_obs_tensor, hvac_action, next_hvac_obs_tensor, hvac_reward)
            if current_step % train_interval == 0:
                for _ in range(2):
                    fan_loss = fan_agent.optimize_model()
                    hvac_loss = hvac_agent.optimize_model()
                    if fan_loss is not None:
                        fan_losses.append(fan_loss)
                    if hvac_loss is not None:
                        hvac_losses.append(hvac_loss)

        state = next_state
        obs_dict = dict(zip(env.get_wrapper_attr("observation_variables"), state))
        obs_dict = append_info_and_time_to_dict_pmv(obs_dict, info, current_step, timesteps_per_hour)
        obs_dict = append_fan_speed_to_dict(
            obs_dict, all_action_map[combined_action][3], all_action_map[combined_action][2]
        )
        obs_dict = append_raw_action_to_dict(obs_dict, combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)

        total_fan_reward += fan_reward.item()
        total_hvac_reward += hvac_reward.item()
        prev_fan_action = fan_action.item()
        prev_hvac_action = hvac_action.item()

        if done:
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        current_step += 1

    env.close()
    avg_fan_loss = sum(fan_losses) / len(fan_losses) if fan_losses else 0.0
    avg_hvac_loss = sum(hvac_losses) / len(hvac_losses) if hvac_losses else 0.0
    return total_fan_reward, total_hvac_reward, avg_fan_loss, avg_hvac_loss, all_obs_dict


def cleanup_simulation_folders():
    eplus_folders = glob.glob("Eplus-env-*")
    if eplus_folders:
        print(f"\nCleaning up {len(eplus_folders)} simulation folders...")
        for folder in eplus_folders:
            try:
                shutil.rmtree(folder)
            except Exception as exc:
                print(f"Warning: Could not remove {folder}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Standalone MADQN PMV Training (No WandB)")
    add_common_train_args(parser, season_choices=["hot", "cool", "mixed", "ankara"])
    parser.add_argument("--co2-weight", type=float, default=0.8)
    parser.add_argument("--pmv-weight", type=float, default=0.4)
    parser.add_argument("--switching-penalty", type=float, default=0.0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
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
    run_name = build_train_run_name(
        algo="madqn_pmv",
        env_id=args.env_id,
        season=args.season,
        episodes=args.episodes,
        use_semantic=args.use_semantic,
        semantic_mode=args.semantic_mode,
        semantic_model_path=semantic_model_path,
        semantic_latent_dim=semantic_latent_dim if args.use_semantic else None,
        run_tag=args.run_tag,
        timestamp=timestamp,
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # PMV MADQN obs sizes from madqn_train_fully_cooperative.py:
    # WindowFan: 5, HVAC: 9
    fan_state_size = 5 if not args.use_semantic else (semantic_latent_dim if semantic_mode == "latent" else 5 + semantic_latent_dim)
    hvac_state_size = 9 if not args.use_semantic else (semantic_latent_dim if semantic_mode == "latent" else 9 + semantic_latent_dim)
    fan_action_size = 4
    hvac_action_size = 10
    train_interval = 96 * 2
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
        "lr": args.learning_rate,
        "memory_capacity": 2 * 52600,
        "layer_sizes": [256, 256, 256],
    }

    start_date = datetime(1997, 1, 1)
    chunks = generate_chunks(start_date, days_per_chunk, 365, step_size=days_per_chunk, seasons=[args.season])
    train_chunks, val_chunks, _ = balanced_month_sample(chunks, val_chunks_per_month=1, seed=args.seed)
    total_training_steps = len(train_chunks) * args.episodes * steps_per_chunk

    fan_agent = DQNAgent(fan_state_size, fan_action_size, total_training_steps, args.episodes, training_config)
    hvac_agent = DQNAgent(hvac_state_size, hvac_action_size, total_training_steps, args.episodes, training_config)

    print(f"Fan agent: {fan_state_size} -> {fan_action_size}")
    print(f"HVAC agent: {hvac_state_size} -> {hvac_action_size}")

    best_val_total = float("-inf")
    best_fan_model_path = os.path.join(output_dir, "madqn_pmv_fan_best.pth")
    best_hvac_model_path = os.path.join(output_dir, "madqn_pmv_hvac_best.pth")
    episode_metrics = []

    for episode in range(1, args.episodes + 1):
        random.shuffle(train_chunks)
        fan_train_rewards = []
        hvac_train_rewards = []
        fan_losses = []
        hvac_losses = []

        with tqdm(train_chunks, desc=f"Train Ep {episode}", ncols=100) as pbar:
            for chunk in pbar:
                fan_r, hvac_r, fan_l, hvac_l, _ = run_episode(
                    args.env_id,
                    *chunk,
                    "Training",
                    steps_per_chunk,
                    fan_agent,
                    hvac_agent,
                    train_interval,
                    timesteps_per_hour,
                    reward_config,
                    switching_penalty=args.switching_penalty,
                    semantic_provider=semantic_provider,
                    semantic_mode=semantic_mode or "concat",
                )
                fan_train_rewards.append(fan_r)
                hvac_train_rewards.append(hvac_r)
                fan_losses.append(fan_l)
                hvac_losses.append(hvac_l)
                pbar.set_postfix({"FanR": f"{np.mean(fan_train_rewards):.1f}", "HVACR": f"{np.mean(hvac_train_rewards):.1f}"})

        fan_agent.reduce_lr()
        hvac_agent.reduce_lr()

        fan_val_rewards = []
        hvac_val_rewards = []
        val_power = []
        val_pmv_violations = []
        val_co2_violations = []
        with tqdm(val_chunks, desc=f"Val Ep {episode}", ncols=100) as pbar:
            for chunk in pbar:
                fan_r, hvac_r, _, _, obs_dict = run_episode(
                    args.env_id,
                    *chunk,
                    "Validation",
                    steps_per_chunk,
                    fan_agent,
                    hvac_agent,
                    train_interval,
                    timesteps_per_hour,
                    reward_config,
                    switching_penalty=args.switching_penalty,
                    semantic_provider=semantic_provider,
                    semantic_mode=semantic_mode or "concat",
                )
                fan_val_rewards.append(fan_r)
                hvac_val_rewards.append(hvac_r)
                if "total_electricity_HVACs" in obs_dict:
                    hvac_power = sum(obs_dict["total_electricity_HVACs"])
                    window_power = sum(obs_dict.get("window_fan_energies", [0]))
                    val_power.append((hvac_power + window_power) / 3600000)
                if "pmv_violations" in obs_dict:
                    vals = [v for v in obs_dict["pmv_violations"] if v is not None]
                    if vals:
                        val_pmv_violations.append(sum(vals) / len(vals) * 100)
                if "co2_violations" in obs_dict:
                    vals = [v for v in obs_dict["co2_violations"] if v is not None]
                    if vals:
                        val_co2_violations.append(sum(vals) / len(vals) * 100)
                pbar.set_postfix({"FanR": f"{np.mean(fan_val_rewards):.1f}", "HVACR": f"{np.mean(hvac_val_rewards):.1f}"})

        fan_ep_path = os.path.join(output_dir, f"madqn_pmv_fan_ep{episode}.pth")
        hvac_ep_path = os.path.join(output_dir, f"madqn_pmv_hvac_ep{episode}.pth")
        fan_agent.save_model(fan_ep_path)
        hvac_agent.save_model(hvac_ep_path)

        val_total = np.mean(fan_val_rewards) + np.mean(hvac_val_rewards)
        if val_total > best_val_total:
            best_val_total = val_total
            fan_agent.save_model(best_fan_model_path)
            hvac_agent.save_model(best_hvac_model_path)

        avg_train_fan = float(np.mean(fan_train_rewards))
        avg_train_hvac = float(np.mean(hvac_train_rewards))
        avg_val_fan = float(np.mean(fan_val_rewards))
        avg_val_hvac = float(np.mean(hvac_val_rewards))
        avg_train_fan_loss = float(np.mean(fan_losses)) if fan_losses else 0.0
        avg_train_hvac_loss = float(np.mean(hvac_losses)) if hvac_losses else 0.0
        print(
            f"Episode {episode}: "
            f"Train(Fan/HVAC)=({avg_train_fan:.2f}/{avg_train_hvac:.2f}) "
            f"Val(Fan/HVAC)=({avg_val_fan:.2f}/{avg_val_hvac:.2f})"
        )
        episode_metrics.append(
            {
                "episode": episode,
                "train_reward": avg_train_fan + avg_train_hvac,
                "val_reward": avg_val_fan + avg_val_hvac,
                "train_loss": (avg_train_fan_loss + avg_train_hvac_loss) / 2.0,
                "train_reward_fan": avg_train_fan,
                "train_reward_hvac": avg_train_hvac,
                "val_reward_fan": avg_val_fan,
                "val_reward_hvac": avg_val_hvac,
                "train_loss_fan": avg_train_fan_loss,
                "train_loss_hvac": avg_train_hvac_loss,
                "val_power_kwh": float(np.mean(val_power)) if val_power else None,
                "val_pmv_violation_pct": float(np.mean(val_pmv_violations)) if val_pmv_violations else None,
                "val_co2_violation_pct": float(np.mean(val_co2_violations)) if val_co2_violations else None,
            }
        )

    summary = {
        "env_id": args.env_id,
        "season": args.season,
        "episodes": args.episodes,
        "best_validation_total_reward": float(best_val_total),
        "best_fan_model_path": best_fan_model_path,
        "best_hvac_model_path": best_hvac_model_path,
        "output_dir": output_dir,
        "semantic": semantic_args_metadata(args, fallback_model_path=semantic_model_path),
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Training summary saved to: {summary_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)
    print(f"Metrics saved to: {metrics_path}")

    try:
        plot_path = plot_training_curves(
            metrics_path=metrics_path,
            output_dir=output_dir,
            run_label=f"madqn-pmv-{args.env_id}-{args.season}",
        )
        print(f"Training plot saved to: {plot_path}")
    except Exception as exc:
        print(f"Warning: Failed to generate training plot: {exc}")

    cleanup_simulation_folders()


if __name__ == "__main__":
    main()
