#!/usr/bin/env python3
"""
Standalone evaluation script for trained DQN-PMV HVAC models.
Runs WITHOUT WandB - loads a pre-trained model and evaluates on specified season.

Usage:
    cd /storage/Master/ParallelRL/RL-HVAC
    python evaluate_standalone.py --model-path ./standalone_results/.../dqn_best.pth --season hot
    python evaluate_standalone.py --model-path .../dqn_best.pth --use-semantic --season hot
"""

import argparse
import os
import sys
import random
import glob
import shutil
import json
from pathlib import Path
from datetime import datetime
from typing import List
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
    add_common_eval_args,
    add_semantic_args,
    build_eval_run_name,
    build_semantic_provider,
    semantic_args_metadata,
)

# ============================================================================
# Action Mapping (same as training)
# ============================================================================
ALL_ACTION_MAP = {
    0 : [21, 22, 1.0, 0.0], 1 : [21, 22, 1.0, 0.5], 2 : [21, 22, 1.0, 0.75], 3 : [21, 22, 1.0, 1.0],
    4 : [22, 23, 1.0, 0.0], 5 : [22, 23, 1.0, 0.5], 6 : [22, 23, 1.0, 0.75], 7 : [22, 23, 1.0, 1.0],
    8 : [23, 24, 1.0, 0.0], 9 : [23, 24, 1.0, 0.5], 10 : [23, 24, 1.0, 0.75], 11 : [23, 24, 1.0, 1.0],
    12 : [24, 25, 1.0, 0.0], 13 : [24, 25, 1.0, 0.5], 14 : [24, 25, 1.0, 0.75], 15 : [24, 25, 1.0, 1.0],
    16 : [25, 26, 1.0, 0.0], 17 : [25, 26, 1.0, 0.5], 18 : [25, 26, 1.0, 0.75], 19 : [25, 26, 1.0, 1.0],
    20 : [26, 27, 1.0, 0.0], 21 : [26, 27, 1.0, 0.5], 22 : [26, 27, 1.0, 0.75], 23 : [26, 27, 1.0, 1.0],
    24 : [27, 28, 1.0, 0.0], 25 : [27, 28, 1.0, 0.5], 26 : [27, 28, 1.0, 0.75], 27 : [27, 28, 1.0, 1.0],
    28 : [28, 29, 1.0, 0.0], 29 : [28, 29, 1.0, 0.5], 30 : [28, 29, 1.0, 0.75], 31 : [28, 29, 1.0, 1.0],
    32 : [29, 30, 1.0, 0.0], 33 : [29, 30, 1.0, 0.5], 34 : [29, 30, 1.0, 0.75], 35 : [29, 30, 1.0, 1.0],
    36 : [5 , 50, 0.0, 0.0], 37 : [5 , 50, 0.0, 0.5], 38 : [5 , 50, 0.0, 0.75], 39 : [5 , 50, 0.0, 1.0]
}

OBSERVATION_VARIABLES = [
    'month', 'day_of_month', 'hour', 'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature', 'air_humidity', 
    'people_occupant', 'air_co2', 'window_fan_energy', 'pmv', 'ppd', 'total_electricity_HVAC',
]

def get_agent_observation(observation: torch.Tensor, action: int) -> List[float]:
    """Extract relevant features for the combined agent."""
    if isinstance(observation, torch.Tensor):
        observation = observation.squeeze().tolist()
    
    obs_dict = dict(zip(OBSERVATION_VARIABLES, observation))
    obs_dict = append_fan_speed_to_dict(obs_dict, ALL_ACTION_MAP[action][3], ALL_ACTION_MAP[action][2])
    
    MONTH_MIN, MONTH_MAX = 1.0, 12.0
    DOM_MIN, DOM_MAX = 1.0, 31.0
    month_raw = int(round(obs_dict['month'] * (MONTH_MAX - MONTH_MIN) + MONTH_MIN))
    dom_raw = int(round(obs_dict['day_of_month'] * (DOM_MAX - DOM_MIN) + DOM_MIN))
    
    try:
        dt = datetime(YEAR, month_raw, dom_raw)
        obs_dict['weekday'] = dt.weekday() / 6.0
    except ValueError:
        obs_dict['weekday'] = 0.0
    
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


def run_evaluation(env_id, start_date, end_date, season, steps_per_chunk, agent, 
                   timesteps_per_hour, reward_config, semantic_provider=None, semantic_mode: str = "concat"):
    """Run a single evaluation episode (greedy action selection only)."""
    env = create_environment(
        env_id, start_date, end_date, season, 
        CO2andPMVReward, episode_type="Validation",
        timesteps_per_hour=timesteps_per_hour, reward_kwargs=reward_config
    )
    
    state, info = env.reset()
    combined_action = 20
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    current_step = 1
    total_reward = 0
    all_obs_dict = {}
    
    while current_step < steps_per_chunk:
        normalized_state = torch.tensor(
            min_max_normalize(state, obs_mins_pmv, obs_maxs_pmv), 
            dtype=torch.float32, device=device
        )
        
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
        
        # Greedy action (no exploration)
        action = agent.choose_greedy_action(combined_obs_tensor)
        combined_action = action.item()
        
        next_state, reward, truncated, terminated, info = env.step(combined_action)
        done = terminated or truncated
        
        state = next_state
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), state))
        obs_dict = append_info_and_time_to_dict_pmv(obs_dict, info, current_step, timesteps_per_hour)
        obs_dict = append_fan_speed_to_dict(obs_dict, ALL_ACTION_MAP[combined_action][3], ALL_ACTION_MAP[combined_action][2])
        obs_dict = append_raw_action_to_dict(obs_dict, combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        total_reward += reward
        if done:
            state, info = env.reset()
            combined_action = 20
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            current_step += 1
            continue
        current_step += 1
    
    env.close()
    return total_reward, all_obs_dict


def main():
    parser = argparse.ArgumentParser(description="Standalone Model Evaluation (No WandB)")
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained model (.pth)")
    add_common_eval_args(parser, season_choices=["hot", "cool", "mixed", "ankara"])
    parser.add_argument("--co2-weight", type=float, default=0.5)
    parser.add_argument("--pmv-weight", type=float, default=0.4)
    add_semantic_args(parser)
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Configuration
    BASE_STATE_SIZE = 13
    semantic_latent_dim = 32
        
    ACTION_SIZE = 40
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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = build_eval_run_name(
        algo="dqn",
        env_id=args.env_id,
        season=args.season,
        model_ids=[Path(args.model_path).stem],
        use_semantic=args.use_semantic,
        semantic_mode=args.semantic_mode,
        semantic_model_path=semantic_model_path,
        semantic_latent_dim=semantic_latent_dim if args.use_semantic else None,
        run_tag=args.run_tag,
        timestamp=timestamp,
    )
    output_dir = os.path.join(args.output_dir, run_name)
    os.makedirs(output_dir, exist_ok=True)

    print(f"{'='*60}")
    print(f"Model Evaluation")
    print(f"{'='*60}")
    print(f"📂 Model: {args.model_path}")
    print(f"🌡️  Season: {args.season}")
    print(f"🏢 Environment: {args.env_id}")
    print(f"📁 Output: {output_dir}")
    
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
        "batch_size": 64, "gamma": 0.95, "eps_start": 0.9, "eps_end": 0.01,
        "eps_decay": 5, "tau": 0.005, "lr": 3e-3, "memory_capacity": 2 * 52600,
        "layer_sizes": [256, 256, 256],
    }
    
    # Generate evaluation chunks
    start_date = datetime(1997, 1, 1)
    chunks = generate_chunks(start_date, DAYS_PER_CHUNK, 365, step_size=DAYS_PER_CHUNK, seasons=[args.season])
    _, val_chunks, _ = balanced_month_sample(chunks, val_chunks_per_month=1, seed=args.seed)
    
    print(f"📊 Evaluation chunks: {len(val_chunks)}")
    
    # Initialize and load agent
    agent = DQNAgent(STATE_SIZE, ACTION_SIZE, 1, 1, training_config)
    agent.load_model(args.model_path)
    print(f"✅ Model loaded successfully")
    print(f"   Device: {device}")
    
    # Run evaluation
    rewards = []
    power_list = []
    pmv_violations = []
    co2_violations = []
    
    print(f"\n{'='*60}")
    print(f"Running Evaluation...")
    print(f"{'='*60}")
    
    with tqdm(val_chunks, desc="Evaluating", ncols=100) as pbar:
        for chunk in pbar:
            reward, obs_dict = run_evaluation(
                args.env_id, *chunk, STEPS_PER_CHUNK, agent,
                TIMESTEPS_PER_HOUR, reward_config,
                semantic_provider=semantic_provider,
                semantic_mode=semantic_mode
            )
            rewards.append(reward)
            
            # Calculate metrics
            if 'total_electricity_HVACs' in obs_dict:
                hvac_power = sum(obs_dict['total_electricity_HVACs'])
                window_power = sum(obs_dict.get('window_fan_energies', [0]))
                power_list.append((hvac_power + window_power) / 3600000)  # kWh
            
            if 'pmv_violations' in obs_dict:
                pmv_viol = [v for v in obs_dict['pmv_violations'] if v is not None]
                if pmv_viol:
                    pmv_violations.append(sum(pmv_viol) / len(pmv_viol) * 100)
            
            if 'co2_violations' in obs_dict:
                co2_viol = [v for v in obs_dict['co2_violations'] if v is not None]
                if co2_viol:
                    co2_violations.append(sum(co2_viol) / len(co2_viol) * 100)
            
            pbar.set_postfix({"R": f"{np.mean(rewards):.1f}"})
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 Evaluation Results")
    print(f"{'='*60}")
    print(f"  Average Reward:      {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
    if power_list:
        print(f"  Average Power (kWh): {np.mean(power_list):.2f} ± {np.std(power_list):.2f}")
    if pmv_violations:
        print(f"  PMV Violation (%):   {np.mean(pmv_violations):.2f} ± {np.std(pmv_violations):.2f}")
    if co2_violations:
        print(f"  CO2 Violation (%):   {np.mean(co2_violations):.2f} ± {np.std(co2_violations):.2f}")
    print(f"{'='*60}")
    
    # Save results
    results = {
        "model_path": args.model_path,
        "season": args.season,
        "env_id": args.env_id,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "avg_power_kwh": float(np.mean(power_list)) if power_list else None,
        "pmv_violation_pct": float(np.mean(pmv_violations)) if pmv_violations else None,
        "co2_violation_pct": float(np.mean(co2_violations)) if co2_violations else None,
    }
    
    results_path = os.path.join(output_dir, "results.json")
    results["semantic"] = semantic_args_metadata(args, fallback_model_path=semantic_model_path)
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n💾 Results saved to: {results_path}")
    
    # Generate visualizations
    print(f"\n📊 Generating visualizations...")
    generate_plots(obs_dict, output_dir, args.season)
    print(f"✅ Plots saved to: {output_dir}")
    
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


def generate_plots(obs_dict, output_dir, season):
    """Generate evaluation plots."""
    import matplotlib.pyplot as plt
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle(f'HVAC Control Evaluation - {season.upper()} Season', fontsize=14, fontweight='bold')
    
    timesteps = range(len(obs_dict.get('air_temperatures', [])))
    
    # 1. Temperature Plot
    ax1 = axes[0, 0]
    if 'air_temperatures' in obs_dict and 'outdoor_temperatures' in obs_dict:
        ax1.plot(timesteps, obs_dict['air_temperatures'], label='Indoor', color='#e74c3c', linewidth=1)
        ax1.plot(timesteps, obs_dict['outdoor_temperatures'], label='Outdoor', color='#3498db', linewidth=1, alpha=0.7)
        ax1.axhline(y=23, color='green', linestyle='--', alpha=0.5, label='Comfort Range')
        ax1.axhline(y=26, color='green', linestyle='--', alpha=0.5)
        ax1.fill_between(timesteps, 23, 26, alpha=0.1, color='green')
        ax1.set_ylabel('Temperature (°C)')
        ax1.set_title('Indoor vs Outdoor Temperature')
        ax1.legend(loc='upper right', fontsize=8)
    
    # 2. PMV Plot
    ax2 = axes[0, 1]
    if 'pmvs' in obs_dict:
        pmvs = np.array(obs_dict['pmvs'])
        occupants = np.array(obs_dict.get('people_occupants', [1]*len(pmvs)))
        # Only show PMV when occupied
        pmv_masked = np.where(occupants > 0, pmvs, np.nan)
        ax2.plot(timesteps, pmv_masked, color='#9b59b6', linewidth=1)
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Comfort Bounds (±0.5)')
        ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.5)
        ax2.fill_between(timesteps, -0.5, 0.5, alpha=0.1, color='green')
        ax2.set_ylabel('PMV Index')
        ax2.set_title('Thermal Comfort (PMV)')
        ax2.set_ylim(-3, 3)
        ax2.legend(loc='upper right', fontsize=8)
    
    # 3. Power Consumption
    ax3 = axes[1, 0]
    if 'total_electricity_HVACs' in obs_dict:
        hvac_power = np.array(obs_dict['total_electricity_HVACs']) / 1000  # kW
        fan_power = np.array(obs_dict.get('window_fan_energies', [0]*len(hvac_power))) / 1000
        ax3.fill_between(timesteps, 0, hvac_power, alpha=0.7, label='HVAC', color='#e74c3c')
        ax3.fill_between(timesteps, hvac_power, hvac_power + fan_power, alpha=0.7, label='Fan', color='#3498db')
        ax3.set_ylabel('Power (kW)')
        ax3.set_title('Energy Consumption')
        ax3.legend(loc='upper right', fontsize=8)
    
    # 4. CO2 Levels
    ax4 = axes[1, 1]
    if 'air_co2s' in obs_dict:
        co2 = obs_dict['air_co2s']
        ax4.plot(timesteps, co2, color='#27ae60', linewidth=1)
        ax4.axhline(y=800, color='orange', linestyle='--', alpha=0.7, label='Threshold (800 ppm)')
        ax4.axhline(y=1000, color='red', linestyle='--', alpha=0.5, label='High (1000 ppm)')
        ax4.set_ylabel('CO2 (ppm)')
        ax4.set_title('Air Quality (CO2)')
        ax4.legend(loc='upper right', fontsize=8)
    
    # 5. Agent Actions
    ax5 = axes[2, 0]
    if 'raw_actions' in obs_dict:
        actions = obs_dict['raw_actions']
        ax5.scatter(timesteps, actions, s=1, alpha=0.5, c='#8e44ad')
        ax5.set_ylabel('Action Index')
        ax5.set_title('Agent Actions Over Time')
        ax5.set_ylim(-1, 41)
    
    # 6. Occupancy
    ax6 = axes[2, 1]
    if 'people_occupants' in obs_dict:
        occupants = obs_dict['people_occupants']
        ax6.fill_between(timesteps, 0, occupants, alpha=0.7, color='#f39c12')
        ax6.set_ylabel('Occupants')
        ax6.set_title('Room Occupancy')
        ax6.set_xlabel('Timestep')
    
    # Set x-label for bottom plots
    axes[2, 0].set_xlabel('Timestep')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'evaluation_plots.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create a summary plot
    create_summary_plot(obs_dict, output_dir, season)


def create_summary_plot(obs_dict, output_dir, season):
    """Create a single-page summary visualization."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate hourly averages for cleaner visualization
    if 'air_temperatures' in obs_dict:
        temps = np.array(obs_dict['air_temperatures'])
        hours = len(temps) // 4 if len(temps) > 4 else 1
        hourly_temps = [np.mean(temps[i*4:(i+1)*4]) for i in range(hours)]
        
        ax.plot(range(hours), hourly_temps, label='Indoor Temp', color='#e74c3c', linewidth=2)
        ax.axhspan(23, 26, alpha=0.2, color='green', label='Comfort Zone')
        ax.set_xlabel('Hours')
        ax.set_ylabel('Temperature (°C)')
        ax.set_title(f'Hourly Temperature Profile - {season.upper()} Season')
        ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'temperature_profile.png'), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == "__main__":
    main()
