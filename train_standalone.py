#!/usr/bin/env python3
"""
Standalone training script for DQN-PMV HVAC control.
Runs WITHOUT WandB - all configs are set locally.

Usage:
    cd /storage/Master/ParallelRL/RL-HVAC
    python train_standalone.py --episodes 2 --season hot
    python train_standalone.py --episodes 2 --season hot --use-semantic  # With GNN

Requirements:
    - EnergyPlus installed at /opt/EnergyPlus (or set EPLUS_PATH)
    - sinergym package installed (from ../sinergym-a403)
"""

import argparse
import os
import sys
import random
import glob
import shutil
import json
from datetime import datetime
from typing import List, Dict, Any
from tqdm import tqdm

import numpy as np
import torch

# Add paths
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import sinergym and set up environment
import sinergym
from sinergym.utils.constants import YEAR

# Import local modules
from algorithms.dqn.dqn import DQNAgent, device
from environments.reward import CO2andPMVReward
from environments.environment import create_environment
from utils.dataset import generate_chunks, balanced_month_sample
from utils.experiment_utils import (
    add_observation, update_combined_dict, 
    min_max_normalize, obs_mins_pmv, obs_maxs_pmv,
    append_info_and_time_to_dict_pmv, append_fan_speed_to_dict,
    append_raw_action_to_dict, save_observations_to_csv
)
from utils.standalone_common import (
    add_common_train_args,
    add_semantic_args,
    build_train_run_name,
    build_semantic_provider,
    semantic_args_metadata,
)
from utils.standalone_plots import plot_training_curves

# ============================================================================
# Action Mapping
# ============================================================================
ALL_ACTION_MAP = {
    0 : [21, 22, 1.0, 0.0],
    1 : [21, 22, 1.0, 0.5],
    2 : [21, 22, 1.0, 0.75],
    3 : [21, 22, 1.0, 1.0],
    4 : [22, 23, 1.0, 0.0],
    5 : [22, 23, 1.0, 0.5],
    6 : [22, 23, 1.0, 0.75],
    7 : [22, 23, 1.0, 1.0],
    8 : [23, 24, 1.0, 0.0],
    9 : [23, 24, 1.0, 0.5],
    10 : [23, 24, 1.0, 0.75],
    11 : [23, 24, 1.0, 1.0],
    12 : [24, 25, 1.0, 0.0],
    13 : [24, 25, 1.0, 0.5],
    14 : [24, 25, 1.0, 0.75],
    15 : [24, 25, 1.0, 1.0],
    16 : [25, 26, 1.0, 0.0],
    17 : [25, 26, 1.0, 0.5],
    18 : [25, 26, 1.0, 0.75],
    19 : [25, 26, 1.0, 1.0],
    20 : [26, 27, 1.0, 0.0],
    21 : [26, 27, 1.0, 0.5],
    22 : [26, 27, 1.0, 0.75],
    23 : [26, 27, 1.0, 1.0],
    24 : [27, 28, 1.0, 0.0],
    25 : [27, 28, 1.0, 0.5],
    26 : [27, 28, 1.0, 0.75],
    27 : [27, 28, 1.0, 1.0],
    28 : [28, 29, 1.0, 0.0],
    29 : [28, 29, 1.0, 0.5],
    30 : [28, 29, 1.0, 0.75],
    31 : [28, 29, 1.0, 1.0],
    32 : [29, 30, 1.0, 0.0],
    33 : [29, 30, 1.0, 0.5],
    34 : [29, 30, 1.0, 0.75],
    35 : [29, 30, 1.0, 1.0],
    36 : [5 , 50, 0.0, 0.0],   # HVAC OFF
    37 : [5 , 50, 0.0, 0.5],
    38 : [5 , 50, 0.0, 0.75],
    39 : [5 , 50, 0.0, 1.0]
}

OBSERVATION_VARIABLES = [
    'month', 'day_of_month', 'hour',
    'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature',
    'air_humidity', 'people_occupant', 'air_co2',
    'window_fan_energy', 'pmv', 'ppd', 'total_electricity_HVAC',
]

# ============================================================================
# Helper Functions  
# ============================================================================
def get_agent_observation(observation: torch.Tensor, action: int) -> List[float]:
    """Extract relevant features for the combined agent."""
    if isinstance(observation, torch.Tensor):
        observation = observation.squeeze().tolist()
    
    obs_dict = dict(zip(OBSERVATION_VARIABLES, observation))
    obs_dict = append_fan_speed_to_dict(
        obs_dict,
        ALL_ACTION_MAP[action][3],
        ALL_ACTION_MAP[action][2]
    )
    
    # De-normalize calendar fields for weekday calculation
    MONTH_MIN, MONTH_MAX = 1.0, 12.0
    DOM_MIN, DOM_MAX = 1.0, 31.0
    
    month_raw = int(round(obs_dict['month'] * (MONTH_MAX - MONTH_MIN) + MONTH_MIN))
    dom_raw = int(round(obs_dict['day_of_month'] * (DOM_MAX - DOM_MIN) + DOM_MIN))
    
    try:
        dt = datetime(YEAR, month_raw, dom_raw)
        obs_dict['weekday'] = dt.weekday() / 6.0
    except ValueError:
        obs_dict['weekday'] = 0.0
    
    # Combined agent observation keys
    keys = ['hour', 'outdoor_temperature', 'outdoor_humidity', 'air_temperature', 
            'air_humidity', 'people_occupant', 'window_fan_speed', 
            'total_electricity_HVAC', 'window_fan_energy', 'air_co2', 
            'weekday', 'pmv', 'ppd']
    
    return [obs_dict[k] for k in keys]


def get_semantic_augmented_observation(observation: torch.Tensor, action: int, 
                                        semantic_provider, raw_state, mode: str = "concat") -> List[float]:
    """Get observation augmented with semantic latent vector from GNN.
    
    Args:
        mode: 'concat' = [base_obs, z_t] or 'latent' = z_t only
    """
    # Build snapshot dict for semantic provider (using raw unnormalized values)
    if isinstance(raw_state, torch.Tensor):
        raw_state = raw_state.squeeze().tolist()
    elif isinstance(raw_state, np.ndarray):
        raw_state = raw_state.tolist()
    
    snapshot = dict(zip(OBSERVATION_VARIABLES, raw_state))
    z_t = semantic_provider.get_state_from_snapshot(snapshot)
    
    if mode == "latent":
        return z_t.tolist()
    else:  # concat
        base_obs = get_agent_observation(observation, action)
        return base_obs + z_t.tolist()


def run_episode(env_id: str, start_date, end_date, season: str, 
                episode_type: str, steps_per_chunk: int, agent: DQNAgent,
                train_interval: int, timesteps_per_hour: int,
                reward_config: dict, switching_penalty: float,
                semantic_provider=None, semantic_mode: str = "concat"):
    """Run a single episode (training or validation).
    
    Args:
        semantic_provider: Optional SemanticStateProvider for GNN augmentation.
        semantic_mode: 'concat' or 'latent' mode for semantic observations.
    """
    
    env = create_environment(
        env_id, start_date, end_date, season, 
        CO2andPMVReward, episode_type=episode_type,
        timesteps_per_hour=timesteps_per_hour, 
        reward_kwargs=reward_config
    )
    
    state, info = env.reset()
    combined_action = 20  # Initial action
    
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    current_step = 1
    total_reward = 0
    loss_list = []
    all_obs_dict = {}
    previous_action = None
    
    while current_step < steps_per_chunk:
        # Normalize state
        normalized_state = torch.tensor(
            min_max_normalize(state, obs_mins_pmv, obs_maxs_pmv), 
            dtype=torch.float32, device=device
        )
        
        # Get observation for agent
        if semantic_provider is not None:
            combined_obs = get_semantic_augmented_observation(
                normalized_state, action=combined_action, 
                semantic_provider=semantic_provider, 
                raw_state=state,
                mode=semantic_mode
            )
        else:
            combined_obs = get_agent_observation(normalized_state, action=combined_action)
        combined_obs_tensor = torch.tensor(combined_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Select action
        if episode_type == "Training":
            action = agent.select_action(combined_obs_tensor)
        else:
            action = agent.choose_greedy_action(combined_obs_tensor)
        
        combined_action = action.item()
        
        # Step environment
        next_state, reward, truncated, terminated, info = env.step(combined_action)
        done = terminated or truncated
        reward = torch.tensor([reward], dtype=torch.float32, device=device)
        
        # Switching penalty
        if previous_action is not None and previous_action != action.item():
            reward -= switching_penalty
        
        # Normalize next state
        normalized_next_state = torch.tensor(
            min_max_normalize(next_state, obs_mins_pmv, obs_maxs_pmv), 
            dtype=torch.float32, device=device
        )
        if semantic_provider is not None:
            next_obs = get_semantic_augmented_observation(
                normalized_next_state, action=combined_action,
                semantic_provider=semantic_provider,
                raw_state=next_state,
                mode=semantic_mode
            )
        else:
            next_obs = get_agent_observation(normalized_next_state, action=combined_action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        # Training update
        if episode_type == "Training":
            agent.store_transition(combined_obs_tensor, action, next_obs_tensor, reward)
            if current_step % train_interval == 0:
                for _ in range(2):
                    loss = agent.optimize_model()
                    if loss is not None:
                        loss_list.append(loss)
        
        state = next_state
        
        # Collect observations for logging
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), state))
        obs_dict = append_info_and_time_to_dict_pmv(obs_dict, info, current_step, timesteps_per_hour)
        obs_dict = append_fan_speed_to_dict(obs_dict, ALL_ACTION_MAP[combined_action][3], ALL_ACTION_MAP[combined_action][2])
        obs_dict = append_raw_action_to_dict(obs_dict, combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        total_reward += reward.item()
        previous_action = action.item()
        
        if done:
            state, info = env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        
        current_step += 1
    
    env.close()
    avg_loss = sum(loss_list) / len(loss_list) if loss_list else 0
    return total_reward, avg_loss, all_obs_dict


def main():
    parser = argparse.ArgumentParser(description="Standalone DQN-PMV Training (No WandB)")
    add_common_train_args(parser, season_choices=["hot", "cool", "mixed", "ankara"])
    parser.add_argument("--co2-weight", type=float, default=0.5)
    parser.add_argument("--pmv-weight", type=float, default=0.4)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--gamma", type=float, default=0.95)
    add_semantic_args(parser)
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # ========================================================================
    # Configuration
    # ========================================================================
    BASE_STATE_SIZE = 13
    semantic_latent_dim = 32
    
    ACTION_SIZE = 40
    TRAIN_INTERVAL = 96 * 2
    TIMESTEPS_PER_HOUR = 4
    DAYS_PER_CHUNK = 8
    STEPS_PER_CHUNK = TIMESTEPS_PER_HOUR * 24 * DAYS_PER_CHUNK
    
    # Initialize semantic provider if requested
    semantic_provider = None
    semantic_mode = None
    semantic_model_path = None
    if args.use_semantic:
        semantic_provider, semantic_mode, semantic_model_path = build_semantic_provider(
            args,
            device=str(device),
            semantic_root="./semantic",
        )
        semantic_latent_dim = semantic_provider.latent_dim

    # Create output directory after semantic init so naming can include GNN details.
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_train_run_name(
        algo="dqn",
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
    print(f"📁 Output directory: {output_dir}")

    # Determine state size based on mode
    if args.use_semantic:
        if args.semantic_mode == "latent":
            STATE_SIZE = semantic_latent_dim
        else:
            STATE_SIZE = BASE_STATE_SIZE + semantic_latent_dim
    else:
        STATE_SIZE = BASE_STATE_SIZE

    if args.use_semantic:
        print(
            f"🧠 Semantic GNN enabled (mode={semantic_mode}, state_dim={STATE_SIZE}, "
            f"checkpoint={semantic_model_path})"
        )
    
    reward_config = {
        'pmv_variables': ['pmv'],
        'co2_variable': 'air_co2',
        'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
        'ac_energy_weight': 1 - args.pmv_weight,
        'fan_energy_weight': 1 - args.co2_weight,
        'co2_weight': args.co2_weight,
        'pmv_weight': args.pmv_weight,
        'lambda_energy': 1/1_600_000,
        'lambda_pmv': 1.0,
        'lambda_co2': 1.0,
        'co2_threshold': 800,
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
    
    # ========================================================================
    # Generate training/validation chunks
    # ========================================================================
    start_date = datetime(1997, 1, 1)
    chunks = generate_chunks(start_date, DAYS_PER_CHUNK, 365, 
                             step_size=DAYS_PER_CHUNK, seasons=[args.season])
    train_chunks, val_chunks, _ = balanced_month_sample(chunks, val_chunks_per_month=1, seed=args.seed)
    
    print(f"📊 Training chunks: {len(train_chunks)}, Validation chunks: {len(val_chunks)}")
    
    # ========================================================================
    # Initialize Agent
    # ========================================================================
    total_training_steps = len(train_chunks) * args.episodes * STEPS_PER_CHUNK
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, total_training_steps, args.episodes, training_config)
    print(f"🤖 Agent initialized: {STATE_SIZE} observations → {ACTION_SIZE} actions")
    print(f"   Device: {device}")
    
    # ========================================================================
    # Training Loop
    # ========================================================================
    best_val_reward = float('-inf')
    episode_metrics = []
    
    for episode in range(1, args.episodes + 1):
        print(f"\n{'='*60}")
        print(f"Episode {episode}/{args.episodes}")
        print(f"{'='*60}")
        
        random.shuffle(train_chunks)
        
        # Training
        train_rewards = []
        train_losses = []
        
        with tqdm(train_chunks, desc="Training", ncols=100) as pbar:
            for chunk in pbar:
                reward, loss, _ = run_episode(
                    args.env_id, *chunk, "Training", STEPS_PER_CHUNK,
                    agent, TRAIN_INTERVAL, TIMESTEPS_PER_HOUR,
                    reward_config, switching_penalty=0.0,
                    semantic_provider=semantic_provider,
                    semantic_mode=semantic_mode
                )
                train_rewards.append(reward)
                train_losses.append(loss)
                pbar.set_postfix({"R": f"{np.mean(train_rewards):.1f}", "L": f"{np.mean(train_losses):.4f}"})
        
        agent.reduce_lr()
        
        # Validation
        val_rewards = []
        val_power = []
        val_pmv_violations = []
        val_co2_violations = []
        
        with tqdm(val_chunks, desc="Validation", ncols=100) as pbar:
            for chunk in pbar:
                reward, _, obs_dict = run_episode(
                    args.env_id, *chunk, "Validation", STEPS_PER_CHUNK,
                    agent, TRAIN_INTERVAL, TIMESTEPS_PER_HOUR,
                    reward_config, switching_penalty=0.0,
                    semantic_provider=semantic_provider,
                    semantic_mode=semantic_mode
                )
                val_rewards.append(reward)
                
                # Calculate power
                if 'total_electricity_HVACs' in obs_dict:
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    window_power = sum(obs_dict.get('window_fan_energies', [0]))
                    val_power.append((hvac_power + window_power) / 3600000)  # to kWh

                if 'pmv_violations' in obs_dict:
                    pmv_viol = [v for v in obs_dict['pmv_violations'] if v is not None]
                    if pmv_viol:
                        val_pmv_violations.append(sum(pmv_viol) / len(pmv_viol) * 100)

                if 'co2_violations' in obs_dict:
                    co2_viol = [v for v in obs_dict['co2_violations'] if v is not None]
                    if co2_viol:
                        val_co2_violations.append(sum(co2_viol) / len(co2_viol) * 100)
                
                pbar.set_postfix({"R": f"{np.mean(val_rewards):.1f}"})
        
        avg_val_reward = np.mean(val_rewards)
        avg_train_reward = np.mean(train_rewards)
        avg_train_loss = np.mean(train_losses)
        
        print(f"\n📈 Episode {episode} Results:")
        print(f"   Train Reward: {avg_train_reward:.2f} | Loss: {avg_train_loss:.4f}")
        print(f"   Val Reward:   {avg_val_reward:.2f}")
        if val_power:
            print(f"   Val Power:    {np.mean(val_power):.2f} kWh")
        if val_pmv_violations:
            print(f"   Val PMV Viol: {np.mean(val_pmv_violations):.2f}%")
        if val_co2_violations:
            print(f"   Val CO2 Viol: {np.mean(val_co2_violations):.2f}%")

        episode_metrics.append(
            {
                "episode": episode,
                "train_reward": float(avg_train_reward),
                "val_reward": float(avg_val_reward),
                "train_loss": float(avg_train_loss),
                "val_power_kwh": float(np.mean(val_power)) if val_power else None,
                "val_pmv_violation_pct": float(np.mean(val_pmv_violations)) if val_pmv_violations else None,
                "val_co2_violation_pct": float(np.mean(val_co2_violations)) if val_co2_violations else None,
            }
        )
        
        # Save model
        model_path = os.path.join(output_dir, f"dqn_ep{episode}.pth")
        agent.save_model(model_path)
        
        if avg_val_reward > best_val_reward:
            best_val_reward = avg_val_reward
            best_model_path = os.path.join(output_dir, "dqn_best.pth")
            agent.save_model(best_model_path)
            print(f"   ✅ New best model saved!")
    
    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"Best validation reward: {best_val_reward:.2f}")
    print(f"Models saved to: {output_dir}")
    print(f"{'='*60}")

    summary = {
        "env_id": args.env_id,
        "season": args.season,
        "episodes": args.episodes,
        "best_validation_reward": float(best_val_reward),
        "output_dir": output_dir,
        "best_model_path": os.path.join(output_dir, "dqn_best.pth"),
        "semantic": semantic_args_metadata(args, fallback_model_path=semantic_model_path),
    }
    summary_path = os.path.join(output_dir, "training_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"💾 Training summary saved to: {summary_path}")

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(episode_metrics, f, indent=2)
    print(f"💾 Metrics saved to: {metrics_path}")

    try:
        plot_path = plot_training_curves(
            metrics_path=metrics_path,
            output_dir=output_dir,
            run_label=f"dqn-{args.env_id}-{args.season}",
        )
        print(f"📊 Training plot saved to: {plot_path}")
    except Exception as exc:
        print(f"Warning: Failed to generate training plot: {exc}")
    
    # Cleanup EnergyPlus simulation folders
    cleanup_simulation_folders()


def cleanup_simulation_folders():
    """Remove EnergyPlus output folders to save disk space."""
    eplus_folders = glob.glob("Eplus-env-*")
    if eplus_folders:
        print(f"\n🧹 Cleaning up {len(eplus_folders)} simulation folders...")
        for folder in eplus_folders:
            try:
                shutil.rmtree(folder)
            except Exception as e:
                print(f"  Warning: Could not remove {folder}: {e}")
        print(f"✅ Cleanup complete!")


if __name__ == "__main__":
    main()
