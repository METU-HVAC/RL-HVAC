import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
from algorithms.dqn.dqn import *
from environments.reward import *
from environments.environment import CO2_AND_TEMP_REWARD_CONFIG
import torch
from sinergym.utils.wrappers import DatetimeWrapper

from environments.environment import create_environment
from utils.dataset import *
from utils.visualization import plot_and_save, plot_csv_data
from utils.experiment_utils import *
from tqdm import tqdm
import wandb
import pandas as pd
import json
reward_log = {
    "timestep": [],
    "ac_fan_energy_term": [],
    "window_fan_energy_term": [],
    "comfort_term": [],
    "co2_term": [],
    "r_total": [],       # if you also want to store the total reward
}
all_action_map = {

}
observation_variables = [
    'month', 'day_of_month', 'hour',
    'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature',
    'air_humidity', 'people_occupant', 'air_co2',
    'window_fan_energy', 'total_electricity_HVAC'
]

raw_observations = []
log_val_dict = []
final_log_dict = []
# Dummy is_summer function
def is_summer(month: float) -> float:
    return float(month in [6.0, 7.0, 8.0])
# Main agent observation extractor
def get_agent_observation_dict_based(agent_name: str, observation: List[float], action: int) -> List[float]:
    if isinstance(observation, torch.Tensor):
        observation = observation.squeeze().tolist()  # flatten if it's (1, N)
    obs_dict = dict(zip(observation_variables, observation))
    # De‐normalize the calendar fields
    #    month_norm in [0..1] → month_raw in [1..12]
    MONTH_MIN, MONTH_MAX = 1.0, 12.0
    DOM_MIN,   DOM_MAX   = 1.0, 31.0

    month_raw = int(round(obs_dict['month'] * (MONTH_MAX - MONTH_MIN) + MONTH_MIN))
    dom_raw   = int(round(obs_dict['day_of_month'] * (DOM_MAX  - DOM_MIN)   + DOM_MIN))
    
    dt = datetime(
        YEAR,
        month_raw,
        dom_raw
    )
    
    obs_dict['weekday'] = dt.weekday() / 6.0
    obs_dict['is_summer'] = is_summer(month_raw)

    keys = ['hour','outdoor_temperature','air_temperature', 'people_occupant', 'is_summer','weekday', 'total_electricity_HVAC']   
    return [obs_dict[k] for k in keys]

def run_simulation(env_id,start_date, end_date, season,episode_type, steps_per_chunk,ac_agent,train_interval,timesteps_per_hour,reward_config):
    env = create_environment(env_id,start_date, end_date,season,LinearReward,episode_type=episode_type,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    
    state, info = env.reset()
    combined_action = 5 #initially the the system is not working
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False

    current_step = 1 
    total_reward = 0
    loss_list = []
    all_obs_dict = {}
    
    normalized_values = []  # Store normalized observations
    
    total_rewards = {"WindowFan": 0.0, "HVAC": 0.0}
    loss_log = {"WindowFan": [], "HVAC": []}
    while current_step < steps_per_chunk:
        
        normalized_state = torch.tensor(min_max_normalize(state,five_zone_obs_mins,five_zone_obs_maxs), dtype=torch.float32, device=device)
        
        #normalized_reduced_state = torch.tensor(normalized_reduced_state, dtype=torch.float32, device=device)
        hvac_obs = get_agent_observation_dict_based("HVAC", normalized_state, action=combined_action)
        
        hvac_obs_tensor = torch.tensor(hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)
        #normalized_values.append(normalized_reduced_state.cpu().numpy())  # Store values for analysis
        if episode_type == "Training":
            hvac_action = ac_agent.select_action(hvac_obs_tensor)
        else:
            hvac_action = ac_agent.choose_greedy_action(hvac_obs_tensor)
            
        # Step in environment
        next_state, reward, terminated, truncated, info = env.step(combined_action)
        done = terminated or truncated
        
        # Get next observations for both agents
        normalized_next_state = torch.tensor(min_max_normalize(next_state,obs_mins,obs_maxs), dtype=torch.float32, device=device)
        next_hvac_obs = get_agent_observation_dict_based("HVAC", normalized_next_state, action=combined_action)    
        reward_tensor = torch.tensor(reward, dtype=torch.float32, device=device).unsqueeze(0)    
        next_hvac_obs_tensor = torch.tensor(next_hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        if episode_type == "Training":
        
            ac_agent.store_transition(hvac_obs_tensor, hvac_action, next_hvac_obs_tensor, reward_tensor)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                hvac_losses = []
                for _ in range(2):
                    hvac_loss = ac_agent.optimize_model()
                    if hvac_loss is not None:
                        hvac_losses.append(hvac_loss)
                if len(hvac_losses) > 0:
                    loss_log["HVAC"].append(sum(hvac_losses) / len(hvac_losses))

        state = next_state
        total_rewards["HVAC"] += reward
                
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), state))

        obs_dict = append_info_and_time_to_dict(obs_dict,info,current_step, timesteps_per_hour)
        obs_dict = append_raw_action_to_dict(obs_dict,combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        
        if done:
            env.reset()

        current_step += 1

    env.close()  # Close the environment after use
    return total_rewards,loss_log,all_obs_dict

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        os.makedirs(config.experiment_save_dir, exist_ok=True)
        seed = 42  # Set seed for reproducibility
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        remove_previous_run_logs()

        train_interval = 96*2 # Train every n steps. Which is 96 steps for 15 minute intervals, which is 24 hours.
        timesteps_per_hour = 4  # 15-minute intervals
        days_per_chunk = 8
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        start_date = datetime(1997, 1, 1)
        total_days = 365
        
        # Generate and split chunks

        mixed_chunks = generate_chunks(start_date, days_per_chunk, total_days, step_size=days_per_chunk, seasons=["mixed"])
        cool_chunks = generate_chunks(start_date, days_per_chunk, total_days,step_size=days_per_chunk,seasons=["cool"])
        hot_chunks = generate_chunks(start_date, days_per_chunk, total_days,step_size=days_per_chunk,seasons=["hot"])

        #train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.1, val_ratio=0.1,seed=seed)
        mixed_train_chunks, _, test_chunks = balanced_month_sample(mixed_chunks, val_chunks_per_month=1, seed=seed)
        cool_train_chunks, _, _ = balanced_month_sample(cool_chunks, val_chunks_per_month=1, seed=seed)
        hot_train_chunks, val_chunks, _ = balanced_month_sample(hot_chunks, val_chunks_per_month=1, seed=seed)

        train_chunks = hot_train_chunks

        num_episodes = config.num_episodes  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        
        temp_weight = config.temp_weight

        lambda_energy = config.lambda_energy
        learning_rate = config.learning_rate
        gamma = config.gamma
        experiment_save_dir = config.experiment_save_dir
        env_id = config.env_id
        layer_sizes = config.layer_sizes
        memory_capacity = config.memory_capacity
        reward_config = {
            'temperature_variables': ['air_temperature'],
            'energy_variables':  ['total_electricity_HVAC'],
            'range_comfort_winter':  (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'summer_start': (6, 1),
            'summer_final': (9, 30),
            'energy_weight': temp_weight,
            'lambda_energy': lambda_energy,
            'lambda_temperature':  1.0
        }
        training_config = {
            "batch_size": 64,
            "gamma": gamma,
            "eps_start": 0.9,
            "eps_end": 0.01,
            "eps_decay": 5,
            "tau": 0.005,
            "lr":learning_rate,
            "memory_capacity": memory_capacity,
            "layer_sizes": layer_sizes,
        }
        ac_agent = DQNAgent(8, 7,total_training_steps,num_episodes,training_config)
        best_ac_val_reward = -float('inf')
        best_ac_model_path = None
        for episode in range(1, num_episodes + 1):
            with tqdm(total=len(train_chunks) + len(val_chunks), 
                    desc=f"Episode {episode}", 
                    ncols=120, 
                    unit="chunk", 
                    leave=True) as pbar:
                # Shuffle train chunks at the start of every episode
                random.shuffle(train_chunks)
                # Training: Full sweep over the shuffled training dataset
                train_total_ac_reward = 0
                train_total_power_list = []
                train_temp_viol_percentage_list = []
                total_ac_loss_list = []
                train_obs_dict = {}
                for train_chunk in train_chunks:
                    obs_dict = {} 
                    rewards,loss_logs,obs_dict = run_simulation(env_id,*train_chunk,
                                                                            "Training", 
                                                                            steps_per_chunk,
                                                                            ac_agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)

                    if loss_logs["HVAC"]:
                        total_ac_loss_list.append(sum(loss_logs["HVAC"]) / len(loss_logs["HVAC"]))
                    else:
                        total_ac_loss_list.append(0.0)  # or 0.0
                    train_obs_dict = update_combined_dict(obs_dict, train_obs_dict)
                    train_total_ac_reward += rewards["HVAC"]
                    
                    #KPI's
                    total_power = sum(obs_dict['total_electricity_HVACs'])
                    
                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    
                    temp_viol_percentage = sum(temp_violations)/len(temp_violations)*100 if temp_violations else 0

                    joules_to_kwh = 1 / 3600000
                    train_total_power_list.append(total_power*joules_to_kwh)
                    train_temp_viol_percentage_list.append(temp_viol_percentage)
                    
                    pbar.set_postfix_str(f"Train Chunk {pbar.n + 1}/{len(train_chunks)}")
                    pbar.update(1)
                    current_training_step += 1

                #An epoch has ended, step the scheduler
                ac_agent.reduce_lr()
                avg_train_ac_reward = (train_total_ac_reward / len(train_chunks))

                train_power_mean = np.mean(train_total_power_list)
                train_power_std = np.std(train_total_power_list)

                train_temp_violation_mean = np.mean(train_temp_viol_percentage_list)
                train_temp_violation_std = np.std(train_temp_viol_percentage_list)
            
                # Validation: Full sweep over the shuffled validation dataset
                val_total_power_list = []
                val_temp_viol_percentage_list = []
                val_obs_dict = {}
                   
                for val_chunk in val_chunks:
                    obs_dict = {}
                    
                    rewards ,loss_logs,obs_dict = run_simulation(env_id,*val_chunk,
                                                                            "Validation", 
                                                                            steps_per_chunk,
                                                                            ac_agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                       
                    
                    
                    val_obs_dict = update_combined_dict(obs_dict, val_obs_dict)
                    append_observations(val_obs_dict,log_val_dict)
                    
                    val_total_ac_reward += rewards["HVAC"]
                    #KPI's
                    total_power = sum(obs_dict['total_electricity_HVACs'])
                    
                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    
                    temp_viol_percentage = sum(temp_violations)/len(temp_violations)*100 if temp_violations else 0
                    
                    val_total_power_list.append(total_power*joules_to_kwh)
                    val_temp_viol_percentage_list.append(temp_viol_percentage)
                    #Timestep temperature and fan speeds
                    inside_temp_levels = val_obs_dict['air_temperatures']
                    outside_temp_levels = val_obs_dict['outdoor_temperatures']
                    raw_actions = val_obs_dict['raw_actions']
                    raw_temp_deviations = val_obs_dict['temp_deviations']

                    pbar.set_postfix_str(f"Val Chunk {pbar.n + 1-len(train_chunks)}/{len(val_chunks)}")
                    pbar.update(1)
                    
                
                ######REWARD _LOGGGING######              
                ac_model_name = "madqn_ac_temp_{:.0f}_lr_{:.0e}".format( temp_weight*100,learning_rate)
                #save_observations_to_csv(log_val_dict, ac_model_name,directory=experiment_save_dir,epoch=episode)
                #Also save model with time
                ac_model_save_dir = f"{experiment_save_dir}/{ac_model_name}_ep{episode}.pth"
                ac_agent.save_model(ac_model_save_dir)
                avg_val_ac_reward = (val_total_ac_reward / len(val_chunks))

                print(
                    f"ACLoss:\t{np.mean(total_ac_loss_list):.3f}\t| "
                    f"Tr.ACR:\t{avg_train_ac_reward:.3f}\t| "
                    f"ValACR:\t{avg_val_ac_reward:.3f}\n")
                
                
                
                if avg_val_ac_reward > best_ac_val_reward:
                    best_ac_val_reward = avg_val_ac_reward
                    best_ac_model_path = ac_model_save_dir
            
                val_power_mean = np.mean(val_total_power_list)
                val_power_std = np.std(val_total_power_list)
                val_temp_violation_mean = np.mean(val_temp_viol_percentage_list)
                val_temp_violation_std = np.std(val_temp_viol_percentage_list)
                
                temp_deviations = [v for v in val_obs_dict['temp_deviations'] if v is not None]

                val_temp_deviation_min = np.min(temp_deviations) if temp_deviations else None
                val_temp_deviation_max = np.max(temp_deviations) if temp_deviations else None
                val_temp_deviation_mean = np.mean(temp_deviations) if temp_deviations else None
                val_temp_deviation_std = np.std(temp_deviations) if temp_deviations else None
            
                log_length = len(val_obs_dict['time_labels'])
                #Power
                wandb.log({"train_total_power_kwh_mean": train_power_mean},step=episode * log_length)
                wandb.log({"train_total_power_kwh_std": train_power_std},step=episode * log_length)


                wandb.log({"val_total_power_kwh_mean": val_power_mean},step=episode * log_length)
                wandb.log({"val_total_power_kwh_std": val_power_std},step=episode * log_length)
                
                #Temperature
                wandb.log({"train_temp_violation_mean":train_temp_violation_mean},step=episode * log_length)
                wandb.log({"train_temp_violation_std":train_temp_violation_std},step=episode * log_length)
                wandb.log({"val_temp_violation_mean":val_temp_violation_mean},step=episode * log_length)
                wandb.log({"val_temp_violation_std":val_temp_violation_std},step=episode * log_length)
                
                wandb.log({"val_temp_deviation_mean":val_temp_deviation_mean},step=episode * log_length)
                wandb.log({"val_temp_deviation_std":val_temp_deviation_std},step=episode * log_length)
                wandb.log({"val_temp_deviation_min":val_temp_deviation_min},step=episode * log_length)
                wandb.log({"val_temp_deviation_max":val_temp_deviation_max},step=episode * log_length)
                #Reward
                wandb.log({"train_ac_reward_mean":avg_val_ac_reward},step=episode * log_length)
                wandb.log({"val_ac_reward_mean":avg_val_ac_reward},step=episode * log_length)
            
                # Temperature and Fan Speed plots
                for timestep in range(log_length):
                    wandb.log({
                        "val_inside_temperature_timestep": inside_temp_levels[timestep],
                        "val_outside_temperature_timestep": outside_temp_levels[timestep],
                        "val_raw_actions" : raw_actions[timestep],
                        "val_temp_deviation_timestep": raw_temp_deviations[timestep] if raw_temp_deviations[timestep] is not None else 0,
                    }, step=episode * len(inside_temp_levels) + timestep) 
                             
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "Pwr":f"{val_power_mean:.1f}", 
                        "Temp": f"{val_temp_violation_mean:.1f}"
                    }
                )
        #After the training finisheds, we will run the best model once against the validaiton set for the final result.
        if best_ac_model_path:
            print(f"Re-evaluating best models from: {best_ac_model_path}")
            ac_agent.load_model(best_ac_model_path)
            final_val_total_reward = 0
            final_val_ac_total_reward = 0
            final_val_power_list = []
            final_val_temp_viol_list = []
            final_temp_deviations = []
            final_obs_dict = {}
            with tqdm(total=len(val_chunks), 
                        desc=f"Episode {num_episodes + 1} (Final Validation)", 
                        ncols=120, 
                        unit="chunk", 
                        leave=True) as pbar:
                
                for i, val_chunk in enumerate(val_chunks):
                    obs_dict = {} 
                    rewards, _, obs_dict = run_simulation(env_id, *val_chunk,
                                                    "Validation", 
                                                    steps_per_chunk,
                                                    ac_agent,
                                                    train_interval,
                                                    timesteps_per_hour,
                                                    reward_config)
                    final_obs_dict = update_combined_dict(obs_dict, final_obs_dict)
                    append_observations(final_obs_dict, final_log_dict)

                    total_power = sum(obs_dict['total_electricity_HVACs'])
                    joules_to_kwh = 1 / 3600000

                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    temp_devs = [v for v in obs_dict['temp_deviations'] if v is not None]

                    final_temp_deviations.extend(temp_devs)

                    temp_viol_percentage = sum(temp_violations) / len(temp_violations) * 100 if temp_violations else 0

                    final_val_ac_total_reward += rewards["HVAC"]
                    final_val_fan_total_reward += rewards["WindowFan"]
                    final_val_total_reward += (rewards["HVAC"] + rewards["WindowFan"])
                    final_val_power_list.append(total_power * joules_to_kwh)
                    final_val_temp_viol_list.append(temp_viol_percentage)

                    ac_model_name = "madqn_temp_{:.0f}_lr_{:.0e}".format(temp_weight*1000,learning_rate)
                    save_observations_to_csv(final_log_dict, ac_model_name,directory=experiment_save_dir,epoch="final")
                    
                    pbar.set_postfix({
                        "Pwr": f"{np.mean(final_val_power_list):.1f}",
                        "Temp": f"{np.mean(final_val_temp_viol_list):.1f}"
                    })
                    pbar.update(1)
            wandb.log({
                "final_val_reward": (final_val_total_reward / len(val_chunks)),
                "final_val_power_kWh_mean": np.mean(final_val_power_list),
                "final_val_power_kWh_std": np.std(final_val_power_list),
                "final_val_temp_violation_%_mean": np.mean(final_val_temp_viol_list),
                "final_val_temp_violation_%_std": np.std(final_val_temp_viol_list),
                "final_temp_deviation_mean": np.mean(final_temp_deviations),
                "final_temp_deviation_std": np.std(final_temp_deviations),

            })