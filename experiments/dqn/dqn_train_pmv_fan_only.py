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

observation_variables = [
    'month', 'day_of_month', 'hour',
    'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature',
    'air_humidity', 'people_occupant', 'air_co2',
    'window_fan_energy', 'pmv','ppd','total_electricity_HVAC',
]
all_action_map = {
    0 : [5 , 50, 0.0, 0.0],
    1 : [5 , 50, 0.0, 0.5],
    2 : [5 , 50, 0.0, 0.75],
    3 : [5 , 50, 0.0, 1.0]
}

SUMMER_START = (1, 1)  # March 1st
SUMMER_END = (12, 30)  # October 30th
def is_summer_by_month(current_month: int, summer_start: tuple, summer_end: tuple) -> bool:
    start_month = summer_start[0]
    end_month = summer_end[0]

    return start_month <= current_month <= end_month

def get_agent_observation_dict_based(agent_name: str, observation: List[float], action: int) -> List[float]:
    if isinstance(observation, torch.Tensor):
        observation = observation.squeeze().tolist()  # flatten if it's (1, N)
    obs_dict = dict(zip(observation_variables, observation))
    obs_dict = append_fan_speed_to_dict(
        obs_dict,
        all_action_map[action][3],
        all_action_map[action][2]
    )
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
    
    if agent_name == "WindowFan":
        keys = ['hour', 'air_co2', 'window_fan_energy', 'people_occupant','weekday']
    elif agent_name == "HVAC":
        keys = ['hour','outdoor_temperature','air_temperature', 'people_occupant', 'window_fan_speed','weekday', 'total_electricity_HVAC','pmv','ppd']
    elif agent_name == "CombinedAgent":
        keys = ['hour', 'outdoor_temperature', 'outdoor_humidity', 'air_temperature', 'people_occupant', 
                'window_fan_speed', 'total_electricity_HVAC','window_fan_energy','air_co2','weekday','pmv','ppd']
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return [obs_dict[k] for k in keys]
raw_observations = []
log_val_dict = []
final_log_dict = []
def run_simulation(env_id,start_date, end_date, season,episode_type, steps_per_chunk,agent,train_interval,timesteps_per_hour,reward_config,switching_penalty):
    env = create_environment(env_id,start_date, end_date,season,CO2andPMVReward,episode_type=episode_type,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    
    state, info = env.reset()
    combined_action = 0 #initially the the system is not working

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False

    current_step = 1 
    total_reward = 0
    loss_list = []
    all_obs_dict = {}

    previous_action = None
    while current_step < steps_per_chunk:

        normalized_state = torch.tensor(min_max_normalize(state,obs_mins_pmv,obs_maxs_pmv), dtype=torch.float32, device=device)
        
        combined_obs = get_agent_observation_dict_based("WindowFan", normalized_state, action=combined_action)
        combined_obs_tensor = torch.tensor(combined_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        if episode_type == "Training":
            action = agent.select_action(combined_obs_tensor)
        else:
            action = agent.choose_greedy_action(combined_obs_tensor) 
        combined_action = action.item()
        next_state, reward, truncated, terminated, info = env.step(combined_action)
        done = terminated or truncated
        reward = torch.tensor([reward],dtype=torch.float32, device=device)
        if previous_action is not None:
            hvac_switched = previous_action != action.item()
            hvac_penalty = torch.tensor([-switching_penalty if hvac_switched else 0.0], dtype=torch.float32, device=device)
        else:
            hvac_penalty = torch.tensor([0.0], dtype=torch.float32, device=device)
        reward += hvac_penalty    
        #Switching penalty
        # if previous_action != action.item():
        #     reward -= 0.1
        # previous_action = action.item()
        normalized_next_state = torch.tensor(min_max_normalize(next_state,obs_mins_pmv,obs_maxs_pmv), dtype=torch.float32, device=device)
        next_obs = get_agent_observation_dict_based("WindowFan", normalized_next_state, action=combined_action)
        next_obs_tensor = torch.tensor(next_obs, dtype=torch.float32, device=device).unsqueeze(0)
    
        if episode_type == "Training":
            agent.store_transition(combined_obs_tensor, action, next_obs_tensor, reward)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                # Perform one step of the optimization (on the policy network)
                for _ in range(2):
                    loss = agent.optimize_model()
                    if loss is not None:
                        loss_list.append(loss)
            #next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        state = next_state
            
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), state))

        obs_dict = append_info_and_time_to_dict_pmv(obs_dict,info,current_step, timesteps_per_hour)
        obs_dict = append_fan_speed_to_dict(obs_dict, all_action_map[combined_action][3], all_action_map[combined_action][2])        
        obs_dict = append_raw_action_to_dict(obs_dict,combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        total_reward += reward
        previous_action = action.item()
        if done:
            env.reset()

        current_step += 1
    
    env.close()  # Close the environment after use
    if len(loss_list) > 0:
        loss = sum(loss_list) / len(loss_list)
    else:
        loss = 0
    return total_reward,loss,all_obs_dict

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
                
        state_size =  5 # Adjust based on the size of your observation space
        action_size = 4
        train_interval = 96*2 # Train every n steps
        timesteps_per_hour = 4  # 15-minute intervals
        days_per_chunk = 8
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        start_date = datetime(1997, 1, 1)
        total_days = 365


        # Generate and split chunks
        chunks = generate_chunks(start_date, days_per_chunk, total_days,step_size=days_per_chunk,seasons=[config.train_season])
        #train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.8, val_ratio=0.2,seed=seed)
        train_chunks, val_chunks, test_chunks = balanced_month_sample(chunks, val_chunks_per_month=1, seed=seed)
        num_episodes = config.num_episodes  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)
        total_number_of_test_chunks = len(test_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        total_testing_steps = total_number_of_test_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        

        co2_weight = config.co2_weight
        pmv_weight = config.pmv_weight
        switching_penalty = config.switching_penalty
        fan_energy_weight = 1 - co2_weight
        ac_energy_weight = 1 - pmv_weight
        gamma = config.gamma
        lambda_energy = config.lambda_energy
        learning_rate = config.learning_rate
        experiment_save_dir = config.experiment_save_dir
        env_id = config.env_id
        layer_sizes = config.layer_sizes
        

        reward_config = {
            'pmv_variables': ['pmv'],
            'co2_variable': 'air_co2',
            'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
            'ac_energy_weight': ac_energy_weight,
            'fan_energy_weight': fan_energy_weight,
            'co2_weight': co2_weight,
            'pmv_weight': pmv_weight,
            'lambda_energy': lambda_energy, # 1/100.000
            'lambda_pmv': 1.0,
            'lambda_co2': 1.0,
            'co2_threshold': 800,
        }
        training_config = {
            "batch_size": 64,
            "gamma": gamma,
            "eps_start": 0.9, 
            "eps_end": 0.01,
            "eps_decay": 5,
            "tau": 0.005,
            "lr":learning_rate,
            "memory_capacity": 300000,
            "layer_sizes": layer_sizes,
        }
        agent = DQNAgent(state_size, action_size,total_training_steps,num_episodes,training_config)
        best_val_reward = -float('inf')
        best_model_path = None
        for episode in range(1, num_episodes + 1):
            with tqdm(total=len(train_chunks) + len(val_chunks), 
                    desc=f"Episode {episode}", 
                    ncols=120, 
                    unit="chunk", 
                    leave=True) as pbar:
                # Shuffle train chunks at the start of every episode
                random.shuffle(train_chunks)
                # Training: Full sweep over the shuffled training dataset
                train_total_reward = 0
                train_total_power_list = []
                train_hvac_power_list = []
                train_fan_power_list = []
                train_co2_viol_percentage_list = []
                train_pmv_viol_percentage_list = []
                total_loss_list = []
                train_obs_dict = {}
                for train_chunk in train_chunks:
                    obs_dict = {} 
                    reward,loss,obs_dict = run_simulation(env_id,*train_chunk,
                                                                            "Training", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config,
                                                                            switching_penalty)
                    
                    total_loss_list.append(loss)
                    train_obs_dict = update_combined_dict(obs_dict, train_obs_dict)
                    train_total_reward += reward
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    
                    pmv_violations = [v for v in obs_dict['pmv_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    
                    pmv_viol_percentage = sum(pmv_violations)/len(pmv_violations)*100 if pmv_violations else 0
                    co2_viol_percentage = sum(co2_violations)/len(co2_violations)*100 if co2_violations else 0

                    # Power values are total joules for that timestep, which is 10 minutes(600sec) for now.
                    # Covnert to kWh
                    
                    joules_to_kwh = 1 / 3600000
                    train_total_power_list.append(total_power*joules_to_kwh)
                    train_hvac_power_list.append(hvac_power*joules_to_kwh)
                    train_fan_power_list.append(window_power*joules_to_kwh)
                    train_co2_viol_percentage_list.append(co2_viol_percentage)
                    train_pmv_viol_percentage_list.append(pmv_viol_percentage)

                    pbar.set_postfix_str(f"Train Chunk {pbar.n + 1}/{len(train_chunks)}")
                    pbar.update(1)
                    current_training_step += 1

                #An epoch has ended, step the scheduler
                agent.reduce_lr()
                print(f"Loss for episode {episode}: {np.mean(total_loss_list)}")    
                avg_train_reward = (train_total_reward / len(train_chunks)).item()

                train_power_mean = np.mean(train_total_power_list)
                train_power_std = np.std(train_total_power_list)
                train_hvac_power_mean = np.mean(train_hvac_power_list)
                train_hvac_power_std = np.std(train_hvac_power_list)
                train_fan_power_mean = np.mean(train_fan_power_list)
                train_fan_power_std = np.std(train_fan_power_list)
                train_pmv_violation_mean = np.mean(train_pmv_viol_percentage_list)
                train_pmv_violation_std = np.std(train_pmv_viol_percentage_list)
                train_co2_violation_mean = np.mean(train_co2_viol_percentage_list)
                train_co2_violation_std = np.std(train_co2_viol_percentage_list)
            
                # Validation: Full sweep over the shuffled validation dataset
                val_total_reward = 0
                val_total_power_list = []
                val_hvac_power_list = []
                val_fan_power_list = []
                val_co2_viol_percentage_list = []
                val_pmv_viol_percentage_list = []
                val_obs_dict = {}
                for val_chunk in val_chunks:
                    obs_dict = {}    
                    reward ,loss,obs_dict = run_simulation(env_id,*val_chunk,
                                                                            "Validation", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config,
                                                                            switching_penalty)
                    
                    val_obs_dict = update_combined_dict(obs_dict, val_obs_dict)
                    append_observations(val_obs_dict,log_val_dict)
                    
                    val_total_reward += reward
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    
                    pmv_violations = [v for v in obs_dict['pmv_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    
                    pmv_viol_percentage = sum(pmv_violations)/len(pmv_violations)*100 if pmv_violations else 0
                    co2_viol_percentage = sum(co2_violations)/len(co2_violations)*100 if co2_violations else 0
                    
                    val_total_power_list.append(total_power*joules_to_kwh)
                    val_hvac_power_list.append(hvac_power*joules_to_kwh)
                    val_fan_power_list.append(window_power*joules_to_kwh)
                    val_co2_viol_percentage_list.append(co2_viol_percentage)
                    val_pmv_viol_percentage_list.append(pmv_viol_percentage)
                    #Timestep temperature and fan speeds
                    inside_temp_levels = val_obs_dict['air_temperatures']
                    outside_temp_levels = val_obs_dict['outdoor_temperatures']
                    co2_levels = val_obs_dict['air_co2s']
                    window_fan_speeds = val_obs_dict['window_fan_speeds']
                    ac_fan_speeds = val_obs_dict['ac_fan_speeds']
                    raw_actions = val_obs_dict['raw_actions']
                    raw_pmv_deviations = val_obs_dict['pmv_deviations']
                    raw_co2_deviations = val_obs_dict['co2_deviations']
                    occupants = np.array(val_obs_dict['people_occupants'])
                    pmvs = np.array(val_obs_dict['pmvs'])
                    ppds = np.array(val_obs_dict['ppds'])
                    
                    raw_pmv_values = np.where(occupants > 0, pmvs, 0.0)
                    raw_ppd_values = np.where(occupants > 0, ppds, 5.0)

                    pbar.set_postfix_str(f"Val Chunk {pbar.n + 1-len(train_chunks)}/{len(val_chunks)}")
                    pbar.update(1)
                
                model_name = "dqn_co2_{:.0f}_pmv_{:.0f}_lr_{:.0e}".format( co2_weight*100, pmv_weight*100, learning_rate)
                #save_observations_to_csv(log_val_dict, model_name,directory=experiment_save_dir,epoch=episode)
                #Also save model with time
                model_save_dir = f"{experiment_save_dir}/{model_name}_ep{episode}.pth"
                agent.save_model(model_save_dir)    
                avg_val_reward = (val_total_reward / len(val_chunks)).item()
                if avg_val_reward > best_val_reward:
                    best_val_reward = avg_val_reward
                    best_model_path = model_save_dir

                val_power_mean = np.mean(val_total_power_list)
                val_power_std = np.std(val_total_power_list)
                val_hvac_power_mean = np.mean(val_hvac_power_list)
                val_hvac_power_std = np.std(val_hvac_power_list)
                val_fan_power_mean = np.mean(val_fan_power_list)
                val_fan_power_std = np.std(val_fan_power_list)
                val_pmv_violation_mean = np.mean(val_pmv_viol_percentage_list)
                val_pmv_violation_std = np.std(val_pmv_viol_percentage_list)
                val_co2_violation_mean = np.mean(val_co2_viol_percentage_list)
                val_co2_violation_std = np.std(val_co2_viol_percentage_list)
                
                pmv_deviations = [v for v in val_obs_dict['pmv_deviations'] if v is not None]
                co2_deviations = [v for v in val_obs_dict['co2_deviations'] if v is not None]

                val_pmv_deviation_min = np.min(pmv_deviations) if pmv_deviations else None
                val_pmv_deviation_max = np.max(pmv_deviations) if pmv_deviations else None
                val_pmv_deviation_mean = np.mean(pmv_deviations) if pmv_deviations else None
                val_pmv_deviation_std = np.std(pmv_deviations) if pmv_deviations else None

                val_co2_deviation_min = np.min(co2_deviations) if co2_deviations else None
                val_co2_deviation_max = np.max(co2_deviations) if co2_deviations else None
                val_co2_deviation_mean = np.mean(co2_deviations) if co2_deviations else None
                val_co2_deviation_std = np.std(co2_deviations) if co2_deviations else None
                
                
                log_length = len(val_obs_dict['time_labels'])
                #Power
                wandb.log({"train_total_power_kwh_mean": train_power_mean},step=episode * log_length)
                wandb.log({"train_total_power_kwh_std": train_power_std},step=episode * log_length)
                wandb.log({"train_hvac_power_kwh_mean": train_hvac_power_mean},step=episode * log_length)
                wandb.log({"train_hvac_power_kwh_std": train_hvac_power_std},step=episode * log_length)
                wandb.log({"train_fan_power_kwh_mean": train_fan_power_mean},step=episode * log_length)
                wandb.log({"train_fan_power_kwh_std": train_fan_power_std},step=episode * log_length)
                wandb.log({"val_total_power_kwh_mean": val_power_mean},step=episode * log_length)
                wandb.log({"val_total_power_kwh_std": val_power_std},step=episode * log_length)
                wandb.log({"val_hvac_power_kwh_mean": val_hvac_power_mean},step=episode * log_length)
                wandb.log({"val_hvac_power_kwh_std": val_hvac_power_std},step=episode * log_length)
                wandb.log({"val_fan_power_kwh_mean": val_fan_power_mean},step=episode * log_length)
                wandb.log({"val_fan_power_kwh_std": val_fan_power_std},step=episode * log_length)
                #CO2
                wandb.log({"train_co2_violation_mean":train_co2_violation_mean},step=episode * log_length)
                wandb.log({"train_co2_violation_std":train_co2_violation_std},step=episode * log_length)
                wandb.log({"val_co2_violation_mean":val_co2_violation_mean},step=episode * log_length)
                wandb.log({"val_co2_violation_std":val_co2_violation_std},step=episode * log_length)
                
                wandb.log({"val_co2_deviation_mean":val_co2_deviation_mean},step=episode * log_length)
                wandb.log({"val_co2_deviation_std":val_co2_deviation_std},step=episode * log_length)
                wandb.log({"val_co2_deviation_min":val_co2_deviation_min},step=episode * log_length)
                wandb.log({"val_co2_deviation_max":val_co2_deviation_max},step=episode * log_length)
                #PMV
                wandb.log({"train_pmv_violation_mean":train_pmv_violation_mean},step=episode * log_length)
                wandb.log({"train_pmv_violation_std":train_pmv_violation_std},step=episode * log_length)
                wandb.log({"val_pmv_violation_mean":val_pmv_violation_mean},step=episode * log_length)
                wandb.log({"val_pmv_violation_std":val_pmv_violation_std},step=episode * log_length)
                
                wandb.log({"val_pmv_deviation_mean":val_pmv_deviation_mean},step=episode * log_length)
                wandb.log({"val_pmv_deviation_std":val_pmv_deviation_std},step=episode * log_length)
                wandb.log({"val_pmv_deviation_min":val_pmv_deviation_min},step=episode * log_length)
                wandb.log({"val_pmv_deviation_max":val_pmv_deviation_max},step=episode * log_length)
                #Reward
                wandb.log({"train_reward_mean":avg_train_reward},step=episode * log_length)
                wandb.log({"val_reward_mean":avg_val_reward},step=episode * log_length)
                

                # Temperature and Fan Speed plots
                for timestep in range(log_length):
                    wandb.log({
                        "val_inside_temperature_timestep": inside_temp_levels[timestep],
                        "val_outside_temperature_timestep": outside_temp_levels[timestep],
                        "val_co2_level_timestep": co2_levels[timestep],
                        "val_window_fan_speed_timestep": window_fan_speeds[timestep],
                        "val_ac_fan_speed_timestep": ac_fan_speeds[timestep],
                        "val_raw_actions" : raw_actions[timestep],
                        "val_ppd_timestep": raw_ppd_values[timestep],
                        "val_pmv_timestep": raw_pmv_values[timestep],
                        "val_co2_deviation_timestep": raw_co2_deviations[timestep] if raw_co2_deviations[timestep] is not None else 0,
                    }, step=episode * len(inside_temp_levels) + timestep) 
                             
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "TrR": f"{avg_train_reward:.1f}",
                        "ValR":f"{avg_val_reward:.1f}",
                        "Pwr":f"{val_power_mean:.1f}", 
                        "CO2": f"{val_co2_violation_mean:.1f}",
                        "PMV": f"{val_pmv_violation_mean:.1f}"
                    }
                )
        #After the training finisheds, we will run the best model once against the validaiton set for the final result.
        if best_model_path:
            print(f"Re-evaluating best model from: {best_model_path}")
            agent.load_model(best_model_path)
            final_val_total_reward = 0
            final_val_power_list = []
            final_val_pmv_viol_list = []
            final_val_co2_viol_list = []
            final_pmv_deviations = []
            final_co2_deviations = []
            final_obs_dict = {}
            with tqdm(total=len(val_chunks), 
                        desc=f"Episode {num_episodes + 1} (Final Validation)", 
                        ncols=120, 
                        unit="chunk", 
                        leave=True) as pbar:
                
                for i, val_chunk in enumerate(val_chunks):
                    _, _, obs_dict = run_simulation(env_id, *val_chunk,
                                                    "Validation", 
                                                    steps_per_chunk,
                                                    agent,
                                                    train_interval,
                                                    timesteps_per_hour,
                                                    reward_config,
                                                    switching_penalty)
                    
                    final_obs_dict = update_combined_dict(obs_dict, final_obs_dict)
                    append_observations(final_obs_dict, final_log_dict)

                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    joules_to_kwh = 1 / 3600000

                    pmv_violations = [v for v in obs_dict['pmv_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    pmv_devs = [v for v in obs_dict['pmv_deviations'] if v is not None]
                    co2_devs = [v for v in obs_dict['co2_deviations'] if v is not None]

                    final_pmv_deviations.extend(pmv_devs)
                    final_co2_deviations.extend(co2_devs)

                    pmv_viol_percentage = sum(pmv_violations) / len(pmv_violations) * 100 if pmv_violations else 0
                    co2_viol_percentage = sum(co2_violations) / len(co2_violations) * 100 if co2_violations else 0

                    final_val_total_reward += reward
                    final_val_power_list.append(total_power * joules_to_kwh)
                    final_val_pmv_viol_list.append(pmv_viol_percentage)
                    final_val_co2_viol_list.append(co2_viol_percentage)
                    
                    model_name = "dqn_co2_{:.0f}_pmv_{:.0f}_lr_{:.0e}".format( co2_weight*100, pmv_weight*100, learning_rate)
                    save_observations_to_csv(final_log_dict, model_name, directory=experiment_save_dir, epoch=num_episodes + 1)
                    pbar.set_postfix({
                        "ValR": f"{(final_val_total_reward / (i+1)).item():.1f}",
                        "Pwr": f"{np.mean(final_val_power_list):.1f}",
                        "CO2": f"{np.mean(final_val_co2_viol_list):.1f}",
                        "pmv": f"{np.mean(final_val_pmv_viol_list):.1f}"
                    })
                    pbar.update(1)
            start_step = (num_episodes + 1) * len(final_obs_dict["time_labels"])

            inside_temp_levels = final_obs_dict["air_temperatures"]
            outside_temp_levels = final_obs_dict["outdoor_temperatures"]
            co2_levels = final_obs_dict["air_co2s"]
            window_fan_speeds = final_obs_dict["window_fan_speeds"]
            ac_fan_speeds = final_obs_dict["ac_fan_speeds"]
            raw_actions = final_obs_dict["raw_actions"]
            raw_pmv_deviations = final_obs_dict["pmv_deviations"]
            raw_co2_deviations = final_obs_dict["co2_deviations"]
            occupants = np.array(final_obs_dict["people_occupants"])
            pmvs = np.array(final_obs_dict["pmvs"])
            ppds = np.array(final_obs_dict["ppds"])

            raw_pmv_values = np.where(occupants > 0, pmvs, 0.0)
            raw_ppd_values = np.where(occupants > 0, ppds, 5.0)

            valid_pmvs = raw_pmv_values[raw_pmv_values != 0.0]
            pmv_deviations_from_raw = np.abs(valid_pmvs[np.abs(valid_pmvs) > 0.5]) - 0.5
            pmv_violation_flags = (np.abs(valid_pmvs) > 0.5).astype(int)
            
            pmv_violation_mean = pmv_violation_flags.mean() * 100 
            pmv_violation_std = pmv_violation_flags.std(ddof=0) * 100
            
            valid_ppds = raw_ppd_values[raw_ppd_values != 5.0]
            
            # Log final timestep-level data
            for t in range(len(inside_temp_levels)):
                wandb.log({
                    "final_val_inside_temperature_timestep": inside_temp_levels[t],
                    "final_val_outside_temperature_timestep": outside_temp_levels[t],
                    "final_val_co2_level_timestep": co2_levels[t],
                    "final_val_window_fan_speed_timestep": window_fan_speeds[t],
                    "final_val_ac_fan_speed_timestep": ac_fan_speeds[t],
                    "final_val_raw_actions": raw_actions[t],
                    "final_val_pmv_timestep": raw_pmv_values[t],
                    "final_val_ppd_timestep": raw_ppd_values[t],
                    "final_val_co2_deviation_timestep": raw_co2_deviations[t] if raw_co2_deviations[t] is not None else 0,
                    "final_val_pmv_deviation_timestep": raw_pmv_deviations[t] if raw_pmv_deviations[t] is not None else 0,
                }, step=start_step + t)
            wandb.log({
                "final_val_reward": (final_val_total_reward / len(val_chunks)).item(),

                "final_val_power_kWh_mean": np.mean(final_val_power_list),
                "final_val_power_kWh_std": np.std(final_val_power_list),

                "final_val_pmv_violation_%_mean": pmv_violation_mean,
                "final_val_pmv_violation_%_std": pmv_violation_std,

                "final_val_co2_violation_%_mean": np.mean(final_val_co2_viol_list),
                "final_val_co2_violation_%_std": np.std(final_val_co2_viol_list),

                "final_val_pmv_deviation_mean": np.mean(pmv_deviations_from_raw),
                "final_val_pmv_deviation_std": np.std(pmv_deviations_from_raw),

                "final_val_co2_deviation_mean": np.mean(final_co2_deviations),
                "final_val_co2_deviation_std": np.std(final_co2_deviations),
                
                "final_val_ppd_percentage_mean": np.mean(valid_ppds),
                "final_val_ppd_percentage_std": np.std(valid_ppds),
                
            })

            
# # Create experiment save dir
# train_season = "hot"
# current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
# ENV_ID ="A403medium"             
# unique_experiment_name = f"{train_season}_{ENV_ID}_train_{current_date}"
# experiment_save_dir_name = "results/dqn/" + unique_experiment_name
# if not os.path.exists(experiment_save_dir_name):
#     os.makedirs(experiment_save_dir_name)


# ENV_NAME = f"{ENV_ID}_{train_season}_128_64_MULTI_FAN" 
# ALGORITHM_NAME = "DQN"
# NUM_EPISODES = 8

# name= create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)
# sweep_config = {
#     'method': 'random' ,
#     'name' : name
#     }
# metric = {
#     'name': 'final_val_power_kWh_mean',
#     'goal': 'minimize'   
#     }

# parameters_dict = {
#     'learning_rate': {
#         #'values': [3e-4,1e-3,3e-3]
#         'values': [3e-4]
#     },
#     'lambda_energy': {
#         'values': [1/1_600_000]
#     },
#     # 'energy_weight': {
#     #     'values': [1,2]
#     'co2_weight': {
#         'min': 0.33,
#         'max': 0.50,
#     },
#     'temp_weight': {
#         ## When temp is 1 energy be from 1 to 3. Which means temp weight can be from 0.25 to 0.50
#         'min': 0.20,
#         'max': 0.50,
#     },
#     'experiment_save_dir': {
#         'value': experiment_save_dir_name
#     },
#     'train_season': {
#         'value': train_season
#     },
#     'agent_count': {
#         'value': 10
#     },
#     'num_episodes': {
#         'value': NUM_EPISODES
#     },
#     'env_id': {
#         'value': ENV_ID
#     },
# }
# sweep_config['parameters'] = parameters_dict
# sweep_config['metric'] = metric


# config_save_path = os.path.join(experiment_save_dir_name, "parameters_config.json")
# with open(config_save_path, "w") as f:
#     json.dump(parameters_dict, f, indent=4)
# print(f"Parameters saved to {config_save_path}")

# sweep_id = wandb.sweep(sweep_config, project="A403-Train",entity="mehmetbh")
# wandb.agent(sweep_id, train, count=parameters_dict['agent_count']['value'])

# #Close the agent

# wandb.finish()