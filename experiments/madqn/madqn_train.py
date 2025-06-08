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
# all_action_map = {
#     0: [21.0, 23.0, 1.0, 0.0],
#     1: [21.0, 23.0, 1.0, 1.0],
#     2: [23.0, 26.0, 1.0, 0.0],
#     3: [23.0, 26.0, 1.0, 1.0],
#     4: [5, 50, 0.0, 0.0],
#     5: [5, 50, 0.0, 1.0],
# }
reward_log = {
    "timestep": [],
    "ac_fan_energy_term": [],
    "window_fan_energy_term": [],
    "comfort_term": [],
    "co2_term": [],
    "r_total": [],       # if you also want to store the total reward
}
all_action_map = {
    0 : [21,23,1.0,0.0],
    1 : [21,23,1.0,0.25],
    2 : [21,23,1.0,0.50],
    3 : [21,23,1.0,1.0],
    4 : [23,26,1.0,0.0],
    5 : [23,26,1.0,0.25],
    6 : [23,26,1.0,0.50],
    7 : [23,26,1.0,1.0],
    8 : [5,50,0.0,0.0],
    9 : [5,50,0.0,0.25],
    10 : [5,50,0.0,0.50],
    11 : [5,50,0.0,1.0]
}
fan_map = {
    0: 0.0,   # Off
    1: 0.25,  # Low
    2: 0.5,   # Medium
    3: 1.0    # High
}

hvac_map = {
    0: [5.0, 50.0, 0.0],     # Off
    1: [23.0, 26.0, 1.0],    # Summer
    2: [21.0, 23.0, 1.0]     # Winter
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

# Dummy is_summer function
def is_summer(month: float) -> float:
    return float(month in [6.0, 7.0, 8.0])

# Fan speed appender
def append_fan_speed_to_dict(obs_dict, fan_speed, ac_speed):
    obs_dict['window_fan_speed'] = fan_speed
    obs_dict['ac_fan_speed'] = ac_speed
    return obs_dict

# Main agent observation extractor
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
    obs_dict['is_summer'] = is_summer(month_raw)

    if agent_name == "WindowFan":
        keys = ['hour', 'air_co2', 'window_fan_energy', 'people_occupant','weekday']
    elif agent_name == "HVAC":
        keys = ['hour','outdoor_temperature','air_temperature', 'people_occupant', 'window_fan_speed', 'is_summer','weekday', 'total_electricity_HVAC']
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return [obs_dict[k] for k in keys]

def get_agent_reward(agent_name, info):
    if agent_name == "WindowFan":
        #print(f"WindowFan reward co2 {info["co2_term"]} window energy {info["window_energy_term"]}")
        return info["co2_term"] + info["window_energy_term"]
    elif agent_name == "HVAC":
        #print(f"HVAC reward comfort {info["comfort_term"]} ac energy {info["ac_energy_term"]}")
        return info["comfort_term"] + info["ac_energy_term"]
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
from itertools import product

def generate_combined_action_dict(fan_map: dict, hvac_map: dict) -> dict:
    combined_actions = {}
    idx = 0

    for hvac_action, fan_action in product(hvac_map.values(), fan_map.values()):
        t_min, t_max, hvac_flag = hvac_action
        fan_flag = fan_action
        combined_actions[idx] = [t_min, t_max, hvac_flag, fan_flag]
        idx += 1

    return combined_actions

def get_combined_action_key(fan_action: int, 
                            hvac_action: int, 
                            action_dict: dict, 
                            fan_map: dict, 
                            hvac_map: dict) -> int:
    """
    Returns the key from the action_dict that matches the combination of fan and hvac actions.

    :param fan_action: int, key from fan_map (e.g. 0 or 1)
    :param hvac_action: int, key from hvac_map (e.g. 0, 1, 2)
    :param action_dict: dict, e.g. {0: [21, 23, 1.0, 0.0], ...}
    :param fan_map: dict, e.g. {0: 0.0, 1: 1.0}
    :param hvac_map: dict, e.g. {0: [5, 50, 0.0], 1: [23, 26.0, 1.0], 2: [21, 23, 1.0]}
    :return: int, key from action_dict
    """
    if fan_action not in fan_map:
        raise ValueError(f"Invalid fan_action: {fan_action}")
    if hvac_action not in hvac_map:
        raise ValueError(f"Invalid hvac_action: {hvac_action}")

    # Extract action representation
    fan_flag = fan_map[fan_action]
    t_min, t_max, hvac_flag = hvac_map[hvac_action]

    target_action = [t_min, t_max, hvac_flag, fan_flag]

    # Search for matching action in dict
    for key, value in action_dict.items():
        if value == target_action:
            return key

    raise ValueError("Combined action not found in the provided dictionary.")

def combine_actions(fan_action, hvac_action, action_dict, fan_map, hvac_map):
    fan_speed = fan_map[fan_action]
    t_min, t_max, hvac_flag = hvac_map[hvac_action]
    combined = [t_min, t_max, hvac_flag, fan_speed]

    for key, value in action_dict.items():
        if value == combined:
            return key
    raise ValueError("Invalid fan+hvac action combination")
   
def run_simulation(env_id,start_date, end_date, season,episode_type, steps_per_chunk,fan_agent,ac_agent,train_interval,timesteps_per_hour,reward_config):
    env = create_environment(env_id,start_date, end_date,season,CO2andTemperatureReward,episode_type=episode_type,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    
    state, info = env.reset()
    combined_action = OFF_ACTION = 5 #initially the the system is not working
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
        
        #reduced_state = reduce_state(state)
        normalized_state = torch.tensor(min_max_normalize(state,obs_mins,obs_maxs), dtype=torch.float32, device=device)
        
        #normalized_reduced_state = torch.tensor(normalized_reduced_state, dtype=torch.float32, device=device)
        fan_obs = get_agent_observation_dict_based("WindowFan", normalized_state, action=combined_action)
        hvac_obs = get_agent_observation_dict_based("HVAC", normalized_state, action=combined_action)
        
        fan_obs_tensor = torch.tensor(fan_obs, dtype=torch.float32, device=device).unsqueeze(0)
        hvac_obs_tensor = torch.tensor(hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)
        #normalized_values.append(normalized_reduced_state.cpu().numpy())  # Store values for analysis
        if episode_type == "Training":
            #action = agent.select_action(normalized_reduced_state)  # Epsilon-greedy action for training
            fan_action = fan_agent.select_action(fan_obs_tensor)
            hvac_action = ac_agent.select_action(hvac_obs_tensor)
            
        else:
            #action = agent.choose_greedy_action(normalized_reduced_state)  # Greedy action for validation/testing
            fan_action = fan_agent.choose_greedy_action(fan_obs_tensor)
            hvac_action = ac_agent.choose_greedy_action(hvac_obs_tensor)
            
        combined_action = combine_actions(fan_action.item(), hvac_action.item(),
                                          all_action_map, fan_map, hvac_map)
             
        # Step in environment
        next_state, _, terminated, truncated, info = env.step(combined_action)
        done = terminated or truncated
        
        # Compute rewards per agent
        fan_reward = torch.tensor([get_agent_reward("WindowFan", info)], dtype=torch.float32, device=device)
        hvac_reward = torch.tensor([get_agent_reward("HVAC", info)], dtype=torch.float32, device=device)
        
        # Get next observations for both agents
        normalized_next_state = torch.tensor(min_max_normalize(next_state,obs_mins,obs_maxs), dtype=torch.float32, device=device)
        next_fan_obs = get_agent_observation_dict_based("WindowFan", normalized_next_state, action=combined_action)
        next_hvac_obs = get_agent_observation_dict_based("HVAC", normalized_next_state, action=combined_action)
        
        next_fan_obs_tensor = torch.tensor(next_fan_obs, dtype=torch.float32, device=device).unsqueeze(0)
        next_hvac_obs_tensor = torch.tensor(next_hvac_obs, dtype=torch.float32, device=device).unsqueeze(0)
        
        
        if episode_type == "Training":
        
            fan_agent.store_transition(fan_obs_tensor, fan_action, next_fan_obs_tensor, fan_reward)
            ac_agent.store_transition(hvac_obs_tensor, hvac_action, next_hvac_obs_tensor, hvac_reward)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                fan_loss = fan_agent.optimize_model()
                hvac_loss = ac_agent.optimize_model()
                if fan_loss is not None:
                    loss_log["WindowFan"].append(fan_loss)
                if hvac_loss is not None:
                    loss_log["HVAC"].append(hvac_loss)
            #next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        elif episode_type == "Validation":
            #Log the rewards at each timestep
            ac_fan_energy_reward = info["ac_energy_term"]
            window_fan_energy_reward = info["window_energy_term"]
            temp_reward = info["comfort_term"]
            co2_reward = info["co2_term"]
            
            total_reward = ac_fan_energy_reward + window_fan_energy_reward + temp_reward + co2_reward
            # Append reward_log
            reward_log["timestep"].append(current_step)
            reward_log["ac_fan_energy_term"].append(ac_fan_energy_reward)
            reward_log["window_fan_energy_term"].append(window_fan_energy_reward)
            reward_log["comfort_term"].append(temp_reward)
            reward_log["co2_term"].append(co2_reward)
            reward_log["r_total"].append(total_reward)
        state = next_state
        total_rewards["WindowFan"] += fan_reward.item()
        total_rewards["HVAC"] += hvac_reward.item()
                
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), state))

        obs_dict = append_info_and_time_to_dict(obs_dict,info,current_step, timesteps_per_hour)
       
        obs_dict = append_fan_speed_to_dict(obs_dict, all_action_map[combined_action][3], all_action_map[combined_action][2])
        obs_dict = append_raw_action_to_dict(obs_dict,combined_action)
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        
        
        if done:
            env.reset()

        current_step += 1
    
    # # After loop, analyze normalization
    # normalized_values = np.array(normalized_values)
    # mean = np.mean(normalized_values, axis=0)
    # std = np.std(normalized_values, axis=0)

    # print(f"Mean of normalized observations: {mean}")
    # print(f"Standard deviation of normalized observations: {std}")

    env.close()  # Close the environment after use
    return total_rewards,loss_log,all_obs_dict

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        seed = 42  # Set seed for reproducibility
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        remove_previous_run_logs()
                
        state_size =  10 # Adjust based on the size of your observation space
        action_size = 6  
        train_interval = 200 # Train every n steps
        timesteps_per_hour = 6  # 10-minute intervals
        days_per_chunk = 8
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        start_date = datetime(1997, 1, 1)
        total_days = 365
        plots_dir = "results/plots/setpoint"  # Directory to store plots

        extra_params = {
            'timesteps_per_hour': timesteps_per_hour,
            'runperiod':(1,1,1997,12,3,1997)  # Full year simulation
        }

        # Generate and split chunks
        chunks = generate_chunks(start_date, days_per_chunk, total_days,step_size=days_per_chunk,seasons=[config.train_season])
        #train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.1, val_ratio=0.1,seed=seed)
        train_chunks, val_chunks, test_chunks = balanced_month_sample(chunks, val_chunks_per_month=1, seed=seed)
        num_episodes = config.num_episodes  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)
        total_number_of_test_chunks = len(test_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        total_testing_steps = total_number_of_test_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        
        # total_weight = config.temp_weight + config.co2_weight + config.energy_weight
        # energy_weight = config.energy_weight / total_weight
        # co2_weight = config.co2_weight / total_weight
        # temp_weight = config.temp_weight / total_weight
        #energy_weight, co2_weight, temp_weight = config.normalized_weights_ect
        co2_weight = config.co2_weight
        temp_weight = config.temp_weight
        fan_energy_weight = 1 - co2_weight
        ac_energy_weight = 1 - temp_weight
        lambda_energy = config.lambda_energy
        learning_rate = config.learning_rate
        gamma = config.gamma
        experiment_save_dir = config.experiment_save_dir
        env_id = config.env_id
        
        # if [config.energy_weight, config.co2_weight, config.temp_weight].count(2) != 1:
        #     print(f"Invalid combination: energy={config.energy_weight}, CO2={config.co2_weight}, temp={config.temp_weight}. Only one value should be 2.")
        #     wandb.finish()
        #     return
        reward_config = {
            'temperature_variables': ['air_temperature'],
            'co2_variable': 'air_co2',
            'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'ac_energy_weight': ac_energy_weight,
            'fan_energy_weight': fan_energy_weight,
            'co2_weight': co2_weight,
            'temperature_weight': temp_weight,
            'lambda_energy': lambda_energy, # 1/100.000
            'lambda_temperature': 1.0,
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
            "memory_capacity": 300000
        }
        fan_agent = DQNAgent(5, 4,total_training_steps,training_config)
        ac_agent = DQNAgent(8, 3,total_training_steps,training_config)
        best_ac_val_reward = -float('inf')
        best_fan_val_reward = -float('inf')
        best_ac_model_path = None
        best_fan_model_path = None
        for episode in range(1, num_episodes + 1):
            with tqdm(total=len(train_chunks) + len(val_chunks), 
                    desc=f"Episode {episode}", 
                    ncols=120, 
                    unit="chunk", 
                    leave=True) as pbar:
                # Shuffle train chunks at the start of every episode
                random.shuffle(train_chunks)
                # Training: Full sweep over the shuffled training dataset
                train_total_fan_reward = 0
                train_total_ac_reward = 0
                train_total_power_list = []
                train_hvac_power_list = []
                train_fan_power_list = []
                train_co2_viol_percentage_list = []
                train_temp_viol_percentage_list = []
                total_fan_loss_list = []
                total_ac_loss_list = []
                train_obs_dict = {}
                for train_chunk in train_chunks:
                    obs_dict = {} 
                    rewards,loss_logs,obs_dict = run_simulation(env_id,*train_chunk,
                                                                            "Training", 
                                                                            steps_per_chunk,
                                                                            fan_agent,
                                                                            ac_agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                    total_fan_loss_list.append(sum(loss_logs["WindowFan"])/len(loss_logs["WindowFan"]))
                    total_ac_loss_list.append(sum(loss_logs["HVAC"])/len(loss_logs["HVAC"]))
                    train_obs_dict = update_combined_dict(obs_dict, train_obs_dict)
                    train_total_fan_reward += rewards["WindowFan"]
                    train_total_ac_reward += rewards["HVAC"]
                    
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    
                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    
                    temp_viol_percentage = sum(temp_violations)/len(temp_violations)*100 if temp_violations else 0
                    co2_viol_percentage = sum(co2_violations)/len(co2_violations)*100 if co2_violations else 0

                    # Power values are total joules for that timestep, which is 10 minutes(600sec) for now.
                    # Covnert to kWh
                    

                    joules_to_kwh = 1 / 3600000
                    train_total_power_list.append(total_power*joules_to_kwh)
                    train_hvac_power_list.append(hvac_power*joules_to_kwh)
                    train_fan_power_list.append(window_power*joules_to_kwh)
                    train_co2_viol_percentage_list.append(co2_viol_percentage)
                    train_temp_viol_percentage_list.append(temp_viol_percentage)

                    pbar.set_postfix_str(f"Train Chunk {pbar.n + 1}/{len(train_chunks)}")
                    pbar.update(1)
                    current_training_step += 1

                #An epoch has ended, step the scheduler
                fan_agent.reduce_lr()
                ac_agent.reduce_lr()
                avg_train_fan_reward = (train_total_fan_reward / len(train_chunks))
                avg_train_ac_reward = (train_total_ac_reward / len(train_chunks))

                
                
                train_power_mean = np.mean(train_total_power_list)
                train_power_std = np.std(train_total_power_list)
                train_hvac_power_mean = np.mean(train_hvac_power_list)
                train_hvac_power_std = np.std(train_hvac_power_list)
                train_fan_power_mean = np.mean(train_fan_power_list)
                train_fan_power_std = np.std(train_fan_power_list)
                train_temp_violation_mean = np.mean(train_temp_viol_percentage_list)
                train_temp_violation_std = np.std(train_temp_viol_percentage_list)
                train_co2_violation_mean = np.mean(train_co2_viol_percentage_list)
                train_co2_violation_std = np.std(train_co2_viol_percentage_list)
            
                # Validation: Full sweep over the shuffled validation dataset
                val_total_fan_reward = 0
                val_total_ac_reward = 0
                val_total_power_list = []
                val_hvac_power_list = []
                val_fan_power_list = []
                val_co2_viol_percentage_list = []
                val_temp_viol_percentage_list = []
                val_obs_dict = {}
                reward_log.clear()
                reward_log.update({
                    
                    "timestep": [],
                    "ac_fan_energy_term": [],
                    "window_fan_energy_term": [],
                    "comfort_term": [],
                    "co2_term": [],
                    "r_total": [],
                })    
                for val_chunk in val_chunks:
                    obs_dict = {}
                    
                    rewards ,loss_logs,obs_dict = run_simulation(env_id,*val_chunk,
                                                                            "Validation", 
                                                                            steps_per_chunk,
                                                                            fan_agent,
                                                                            ac_agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                       
                    
                    
                    val_obs_dict = update_combined_dict(obs_dict, val_obs_dict)
                    append_observations(val_obs_dict,log_val_dict)
                    
                    val_total_fan_reward += rewards["WindowFan"]
                    val_total_ac_reward += rewards["HVAC"]
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    
                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    
                    temp_viol_percentage = sum(temp_violations)/len(temp_violations)*100 if temp_violations else 0
                    co2_viol_percentage = sum(co2_violations)/len(co2_violations)*100 if co2_violations else 0
                    
                    val_total_power_list.append(total_power*joules_to_kwh)
                    val_hvac_power_list.append(hvac_power*joules_to_kwh)
                    val_fan_power_list.append(window_power*joules_to_kwh)
                    val_co2_viol_percentage_list.append(co2_viol_percentage)
                    val_temp_viol_percentage_list.append(temp_viol_percentage)
                    #Timestep temperature and fan speeds
                    inside_temp_levels = val_obs_dict['air_temperatures']
                    outside_temp_levels = val_obs_dict['outdoor_temperatures']
                    co2_levels = val_obs_dict['air_co2s']
                    window_fan_speeds = val_obs_dict['window_fan_speeds']
                    ac_fan_speeds = val_obs_dict['ac_fan_speeds']
                    raw_actions = val_obs_dict['raw_actions']
                    raw_temp_deviations = val_obs_dict['temp_deviations']
                    raw_co2_deviations = val_obs_dict['co2_deviations']


                    pbar.set_postfix_str(f"Val Chunk {pbar.n + 1-len(train_chunks)}/{len(val_chunks)}")
                    pbar.update(1)
                    
                ######REWARD _LOGGGING######    
                df = pd.DataFrame(reward_log)

                # Make sure the “validation” directory exists
                os.makedirs("reward_logs", exist_ok=True)

                # Construct a filename that includes episode_id
                csv_path = os.path.join("reward_logs", f"validation_rewards_ep{episode}.csv")

                # Write DataFrame to CSV once, in a single batch
                df.to_csv(csv_path, index=False)

                print(f"[Episode {episode}] Wrote {len(df)} rows to {csv_path}")

                # Now reset the global dict so that the **next** episode starts fresh
                reward_log.clear()
                reward_log.update({
                    "timestep": [],
                    "ac_fan_energy_term": [],
                    "window_fan_energy_term": [],
                    "comfort_term": [],
                    "co2_term": [],
                    "r_total": [],
                })
                ######REWARD _LOGGGING######              
                ac_model_name = "madqn_ac_co2_{:.0f}_temp_{:.0f}_lr_{:.0e}".format(co2_weight*100, temp_weight*100,learning_rate)
                fan_model_name = "madqn_fan_co2_{:.0f}_temp_{:.0f}_lr_{:.0e}".format(co2_weight*100,temp_weight*100, learning_rate)
                #save_observations_to_csv(log_val_dict, ac_model_name,directory=experiment_save_dir,epoch=episode)
                #Also save model with time
                ac_model_save_dir = f"{experiment_save_dir}/{ac_model_name}_ep{episode}.pth"
                fan_model_save_dir = f"{experiment_save_dir}/{fan_model_name}_ep{episode}.pth"
                ac_agent.save_model(ac_model_save_dir)
                fan_agent.save_model(fan_model_save_dir)
                avg_val_ac_reward = (val_total_ac_reward / len(val_chunks))
                avg_val_fan_reward = (val_total_fan_reward / len(val_chunks))

                
                
                print(f"\nFanLoss:\t{np.mean(total_fan_loss_list):.3f}\t| "
                    f"ACLoss:\t{np.mean(total_ac_loss_list):.3f}\t| "
                    f"Tr.FanR:\t{avg_train_fan_reward:.3f}\t| "
                    f"Tr.ACR:\t{avg_train_ac_reward:.3f}\t| "
                    f"ValFanR:\t{avg_val_fan_reward:.3f}\t| "
                    f"ValACR:\t{avg_val_ac_reward:.3f}\n")
                
                
                
                if avg_val_ac_reward > best_ac_val_reward:
                    best_ac_val_reward = avg_val_ac_reward
                    best_ac_model_path = ac_model_save_dir
                if avg_val_fan_reward > best_fan_val_reward:
                    best_fan_val_reward = avg_val_fan_reward
                    best_fan_model_path = fan_model_save_dir
                    

                val_power_mean = np.mean(val_total_power_list)
                val_power_std = np.std(val_total_power_list)
                val_hvac_power_mean = np.mean(val_hvac_power_list)
                val_hvac_power_std = np.std(val_hvac_power_list)
                val_fan_power_mean = np.mean(val_fan_power_list)
                val_fan_power_std = np.std(val_fan_power_list)
                val_temp_violation_mean = np.mean(val_temp_viol_percentage_list)
                val_temp_violation_std = np.std(val_temp_viol_percentage_list)
                val_co2_violation_mean = np.mean(val_co2_viol_percentage_list)
                val_co2_violation_std = np.std(val_co2_viol_percentage_list)
                
                temp_deviations = [v for v in val_obs_dict['temp_deviations'] if v is not None]
                co2_deviations = [v for v in val_obs_dict['co2_deviations'] if v is not None]

                val_temp_deviation_min = np.min(temp_deviations) if temp_deviations else None
                val_temp_deviation_max = np.max(temp_deviations) if temp_deviations else None
                val_temp_deviation_mean = np.mean(temp_deviations) if temp_deviations else None
                val_temp_deviation_std = np.std(temp_deviations) if temp_deviations else None

                val_co2_deviation_min = np.min(co2_deviations) if co2_deviations else None
                val_co2_deviation_max = np.max(co2_deviations) if co2_deviations else None
                val_co2_deviation_mean = np.mean(co2_deviations) if co2_deviations else None
                val_co2_deviation_std = np.std(co2_deviations) if co2_deviations else None
                
                
                # val_co2_deviation_min, val_co2_deviation_max = np.min(co2_deviations), np.max(co2_deviations)
                # val_temp_deviation_min, val_temp_deviation_max = np.min(temp_deviations), np.max(temp_deviations)
                # val_temp_deviation_mean = np.mean(temp_deviations)
                # val_co2_deviation_mean = np.mean(co2_deviations)
                # val_temp_deviation_std = np.std(temp_deviations)
                # val_co2_deviation_std = np.std(co2_deviations)
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
                wandb.log({"train_fan_reward_mean":avg_val_fan_reward},step=episode * log_length)
                wandb.log({"train_ac_reward_mean":avg_val_ac_reward},step=episode * log_length)
                wandb.log({"val_fan_reward_mean":avg_val_fan_reward},step=episode * log_length)
                wandb.log({"val_ac_reward_mean":avg_val_ac_reward},step=episode * log_length)
            
                # Temperature and Fan Speed plots
                for timestep in range(log_length):
                    wandb.log({
                        "val_inside_temperature_timestep": inside_temp_levels[timestep],
                        "val_outside_temperature_timestep": outside_temp_levels[timestep],
                        "val_co2_level_timestep": co2_levels[timestep],
                        "val_window_fan_speed_timestep": window_fan_speeds[timestep],
                        "val_ac_fan_speed_timestep": ac_fan_speeds[timestep],
                        "val_raw_actions" : raw_actions[timestep],
                        "val_temp_deviation_timestep": raw_temp_deviations[timestep] if raw_temp_deviations[timestep] is not None else 0,
                        "val_co2_deviation_timestep": raw_co2_deviations[timestep] if raw_co2_deviations[timestep] is not None else 0,
                    }, step=episode * len(inside_temp_levels) + timestep) 
                             
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "Pwr":f"{val_power_mean:.1f}", 
                        "CO2": f"{val_co2_deviation_mean:.1f}",
                        "Temp": f"{val_temp_violation_mean:.1f}"
                    }
                )
        #After the training finisheds, we will run the best model once against the validaiton set for the final result.
        if best_ac_model_path and best_fan_model_path:
            print(f"Re-evaluating best models from: {best_ac_model_path} and {best_fan_model_path}")
            fan_agent.load_model(best_fan_model_path)
            ac_agent.load_model(best_ac_model_path)
            final_val_total_reward = 0
            final_val_ac_total_reward = 0
            final_val_fan_total_reward = 0
            final_val_power_list = []
            final_val_temp_viol_list = []
            final_val_co2_viol_list = []
            final_temp_deviations = []
            final_co2_deviations = []
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
                                                    fan_agent,
                                                    ac_agent,
                                                    train_interval,
                                                    timesteps_per_hour,
                                                    reward_config)

                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    joules_to_kwh = 1 / 3600000

                    temp_violations = [v for v in obs_dict['temp_violations'] if v is not None]
                    co2_violations = [v for v in obs_dict['co2_violations'] if v is not None]
                    temp_devs = [v for v in obs_dict['temp_deviations'] if v is not None]
                    co2_devs = [v for v in obs_dict['co2_deviations'] if v is not None]

                    final_temp_deviations.extend(temp_devs)
                    final_co2_deviations.extend(co2_devs)

                    temp_viol_percentage = sum(temp_violations) / len(temp_violations) * 100 if temp_violations else 0
                    co2_viol_percentage = sum(co2_violations) / len(co2_violations) * 100 if co2_violations else 0

                    final_val_ac_total_reward += rewards["HVAC"]
                    final_val_fan_total_reward += rewards["WindowFan"]
                    final_val_total_reward += (rewards["HVAC"] + rewards["WindowFan"])
                    final_val_power_list.append(total_power * joules_to_kwh)
                    final_val_temp_viol_list.append(temp_viol_percentage)
                    final_val_co2_viol_list.append(co2_viol_percentage)
                    
                    
                    ac_model_name = "madqn_co2_{:.0f}_temp_{:.0f}_lr_{:.0e}".format(co2_weight*1000, temp_weight*1000,learning_rate)
                    save_observations_to_csv(log_val_dict, ac_model_name,directory=experiment_save_dir,epoch="final")
                    
                    pbar.set_postfix({
                        "Pwr": f"{np.mean(final_val_power_list):.1f}",
                        "CO2": f"{np.mean(final_val_co2_viol_list):.1f}",
                        "Temp": f"{np.mean(final_val_temp_viol_list):.1f}"
                    })
                    pbar.update(1)
            wandb.log({
                "final_val_ac_reward": (final_val_ac_total_reward / len(val_chunks)),
                "final_val_fan_reward": (final_val_fan_total_reward / len(val_chunks)),
                "final_val_reward": (final_val_total_reward / len(val_chunks)),

                "final_val_power_kWh_mean": np.mean(final_val_power_list),
                "final_val_power_kWh_std": np.std(final_val_power_list),

                "final_val_temp_violation_%_mean": np.mean(final_val_temp_viol_list),
                "final_val_temp_violation_%_std": np.std(final_val_temp_viol_list),

                "final_val_co2_violation_%_mean": np.mean(final_val_co2_viol_list),
                "final_val_co2_violation_%_std": np.std(final_val_co2_viol_list),

                "final_temp_deviation_mean": np.mean(final_temp_deviations),
                "final_temp_deviation_std": np.std(final_temp_deviations),

                "final_co2_deviation_mean": np.mean(final_co2_deviations),
                "final_co2_deviation_std": np.std(final_co2_deviations),
            })

            
# # Create experiment save dir
# train_season = "hot"
# current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
# ENV_ID ="A403medium"             
# unique_experiment_name = f"{train_season}_{ENV_ID}_train_{current_date}"
# experiment_save_dir_name = "results/madqn/" + unique_experiment_name
# if not os.path.exists(experiment_save_dir_name):
#     os.makedirs(experiment_save_dir_name)


# ENV_NAME = f"{ENV_ID}_{train_season}_64_64_MULTISPEED_FAN" 
# ALGORITHM_NAME = "MADQN"
# NUM_EPISODES = 10

# name= create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)
# sweep_config = {
#     'method': 'random',
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
#     #     'min': 1,
#     #     'max': 3,
#     # },
#     'gamma': {
#         'min': 0.8,
#         'max': 0.99,
#     },
#     'co2_weight': {
#         'min': 0.30,
#         'max': 0.45,
#     },
#     'temp_weight': {
#         ## When temp is 1 energy be from 1 to 3. Which means temp weight can be from 0.25 to 0.50
#         'min': 0.35,
#         'max': 0.50,
#     },
#     'experiment_save_dir': {
#         'value': experiment_save_dir_name
#     },
#     'train_season': {
#         'value': train_season
#     },
#     'agent_count': {
#         'value': 20
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