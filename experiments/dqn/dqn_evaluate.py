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
from utils.dataset import generate_chunks, split_chunks
from utils.visualization import plot_and_save, plot_csv_data
from utils.experiment_utils import *
from tqdm import tqdm
import wandb
import pandas as pd
import json
## OBSERVATION SPACE 
# {'month': np.float32(7.0), 'day_of_month': np.float32(10.0), 'hour': np.float32(0.0),
#  'outdoor_temperature': np.float32(28.666666), 'outdoor_humidity': np.float32(36.666668), '
# htg_setpoint': np.float32(4.13), 'clg_setpoint': np.float32(50.0), 'air_temperature': np.float32(26.72595), 
# 'air_humidity': np.float32(40.54236), 'people_occupant': np.float32(0.0), 'air_co2': np.float32(456.72827), 
# 'window_fan_energy': np.float32(0.0), 'total_electricity_HVAC': np.float32(0.0)}


raw_observations = []
log_val_dict = []
def run_simulation(env_id,start_date, end_date, season,episode_type, steps_per_chunk,agent,train_interval,timesteps_per_hour,reward_config):
    env = create_environment(env_id,start_date, end_date,season,CO2andTemperatureReward,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    
    state, info = env.reset()
    OFF_ACTION = 6 #initially the the system is not working
    state = append_fan_speed_to_observation(env,state,OFF_ACTION)
    #state = append_rewards_to_observation(env,state,0,0,0)

    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False

    current_step = 1 
    total_reward = 0
    loss_list = []
    all_obs_dict = {}
    
    normalized_values = []  # Store normalized observations


    while current_step < steps_per_chunk:
        
        reduced_state = reduce_state(state)
        normalized_reduced_state = min_max_normalize(reduced_state,reduced_obs_mins,reduced_obs_maxs)
        normalized_reduced_state = torch.tensor(normalized_reduced_state, dtype=torch.float32, device=device)
        
        # normalized_obs = normalize_observation(state,obs_means,obs_stds)
        # normalized_obs = torch.tensor(normalized_obs, dtype=torch.float32, device=device)
        normalized_values.append(normalized_reduced_state.cpu().numpy())  # Store values for analysis
        if episode_type == "Training":
            action = agent.select_action(normalized_reduced_state)  # Epsilon-greedy action for training
        else:
            action = agent.choose_greedy_action(normalized_reduced_state)  # Greedy action for validation/testing

        #np_action = np.array([action], dtype=np.float32)  # Adjust dtype to match environment

        observation, reward, truncated, terminated, info = env.step(action.item())
        observation = append_fan_speed_to_observation(env,observation,action.item())
        #observation = append_rewards_to_observation(env,observation,info['energy_term'],info['comfort_term'],info['co2_term'])
        raw_observations.append(observation)

        #observation, reward, truncated, terminated, info = env.step(action.item())
        done = terminated or truncated
        reward = torch.tensor([reward],dtype=torch.float32, device=device)
        
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        
        if episode_type == "Training":
            # normalized_next_obs = normalize_observation(next_state,obs_means,obs_stds)
            # normalized_next_obs = torch.tensor(normalized_next_obs, dtype=torch.float32, device=device)
            
            
            next_reduced_state = reduce_state(next_state)
            
            normalized_next_reduced_obs = min_max_normalize(next_reduced_state,reduced_obs_mins,reduced_obs_maxs)
            normalized_next_reduced_obs = torch.tensor(normalized_next_reduced_obs, dtype=torch.float32, device=device)
            
            agent.store_transition(normalized_reduced_state, action, normalized_next_reduced_obs, reward)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                # Perform one step of the optimization (on the policy network)
                loss = agent.optimize_model()
                if loss is not None:
                    loss_list.append(loss)
            #next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        state = next_state
            
        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), observation))

        obs_dict = append_info_and_time_to_dict(obs_dict,info,current_step, timesteps_per_hour)
       
        obs_dict = append_fan_speed_to_dict(obs_dict, DEFAULT_A403V3_DISCRETE_FUNCTION(action.item())[3], DEFAULT_A403V3_DISCRETE_FUNCTION(action.item())[2])
        obs_dict = append_raw_action_to_dict(obs_dict,action.item())
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        total_reward += reward
        
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
    if len(loss_list) > 0:
        loss = sum(loss_list) / len(loss_list)
    else:
        loss = 0
    return total_reward,loss,all_obs_dict

def evaluate(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        seed = 42  # Set seed for reproducibility
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        remove_previous_run_logs()
                
        state_size =  10 # Adjust based on the size of your observation space
        action_size = 20  
        train_interval = 200 # Train every n steps
        timesteps_per_hour = 6  # 10-minute intervals
        days_per_chunk = 10
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        start_date = datetime(1997, 1, 1)
        days_per_chunk = 10
        total_days = 365

        # Generate and split chunks
        chunks = generate_chunks(start_date, days_per_chunk, total_days,seasons=[config.eval_season])
        train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.8, val_ratio=0.2,seed=seed)

        num_episodes = config.num_episodes  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)
        total_number_of_test_chunks = len(test_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        total_testing_steps = total_number_of_test_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        
        total_weight = config.temp_weight + config.co2_weight + config.energy_weight
        energy_weight = config.energy_weight / total_weight
        co2_weight = config.co2_weight / total_weight
        temp_weight = config.temp_weight / total_weight
        lambda_energy = config.lambda_energy
        learning_rate = config.learning_rate
        experiment_save_dir = config.experiment_save_dir
        env_id = config.env_id
        reward_config = {
            'temperature_variables': ['air_temperature'],
            'co2_variable': 'air_co2',
            'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'energy_weight': energy_weight,
            'co2_weight': co2_weight,
            'temperature_weight': temp_weight,
            'lambda_energy': lambda_energy, # 1/100.000
            'lambda_temperature': 1.0,
            'lambda_co2': 1.0,
            'co2_threshold': 800,
        }
        training_config = {
            "batch_size": 64,
            "gamma": 0.99,
            "eps_start": 0.9,
            "eps_end": 0.05,
            "eps_decay": 5,
            "tau": 0.005,
            "lr":learning_rate,
            "memory_capacity": 300000
        }
        agent = DQNAgent(state_size, action_size,total_training_steps,training_config)
        
        for episode in range(1, num_episodes + 1):
            with tqdm(total=len(val_chunks), 
                    desc=f"Episode {episode}", 
                    ncols=120, 
                    unit="chunk", 
                    leave=True) as pbar:
                
                # Validation: Full sweep over the shuffled validation dataset
                val_total_reward = 0
                val_total_power_list = []
                val_hvac_power_list = []
                val_fan_power_list = []
                val_co2_viol_percentage_list = []
                val_temp_viol_percentage_list = []
                val_obs_dict = {}
                
                # Load the model
                
                model_name = "dqn_co2_{:.0f}_temp_{:.0f}_energy_{:.0f}_lr_{:.0e}".format(config.co2_weight,config.temp_weight,config.energy_weight,config.learning_rate)
                model_train_dir = f"{config.model_load_path}/{model_name}_ep{episode}.pth"
                agent.load_model(model_train_dir)  # Assumes your DQNAgent has a method 'load' to load the model weights
                
                for val_chunk in val_chunks:
                    obs_dict = {}    
                    reward ,loss,obs_dict = run_simulation(env_id,*val_chunk,
                                                                            "Validation", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                    
                    val_obs_dict = update_combined_dict(obs_dict, val_obs_dict)
                    append_observations(val_obs_dict,log_val_dict)
                    
                    val_total_reward += reward
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    temp_viol_percentage = sum(obs_dict['temp_violations'])/len(obs_dict['temp_violations'])*100
                    co2_viol_percentage = sum(obs_dict['co2_violations'])/len(obs_dict['co2_violations'])*100

                    joules_to_kwh = 1/3600000
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

                    pbar.set_postfix_str(f"val Chunk {pbar.n + 1}/{len(val_chunks)}")
                    pbar.update(1)
                
                model_name = "dqn_co2_{:.0f}_temp_{:.0f}_energy_{:.0f}_lr_{:.0e}".format(config.co2_weight,config.temp_weight,config.energy_weight,config.learning_rate)
                save_observations_to_csv(log_val_dict, model_name,directory=experiment_save_dir,epoch=episode)
                 
                avg_val_reward = (val_total_reward / len(val_chunks)).item()

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

                
                log_length = len(val_obs_dict['time_labels'])
                #Power
                wandb.log({"val_total_power_kwh_mean": val_power_mean},step=episode * log_length)
                wandb.log({"val_total_power_kwh_std": val_power_std},step=episode * log_length)
                wandb.log({"val_hvac_power_kwh_mean": val_hvac_power_mean},step=episode * log_length)
                wandb.log({"val_hvac_power_kwh_std": val_hvac_power_std},step=episode * log_length)
                wandb.log({"val_fan_power_kwh_mean": val_fan_power_mean},step=episode * log_length)
                wandb.log({"val_fan_power_kwh_std": val_fan_power_std},step=episode * log_length)
                #CO2
                wandb.log({"val_co2_violation_mean":val_co2_violation_mean},step=episode * log_length)
                wandb.log({"val_co2_violation_std":val_co2_violation_std},step=episode * log_length)
                #Temperature
                wandb.log({"val_temp_violation_mean":val_temp_violation_mean},step=episode * log_length)
                wandb.log({"val_temp_violation_std":val_temp_violation_std},step=episode * log_length)
                #Reward
                wandb.log({"val_reward_mean":avg_val_reward},step=episode * log_length)

                # Temperature and Fan Speed plots
                for timestep in range(log_length):
                    wandb.log({
                        "val_inside_temperature_timestep": inside_temp_levels[timestep],
                        "val_outside_temperature_timestep": outside_temp_levels[timestep],
                        "val_co2_level_timestep": co2_levels[timestep],
                        "val_window_fan_speed_timestep": window_fan_speeds[timestep],
                        "val_ac_fan_speed_timestep": ac_fan_speeds[timestep],
                        "val_raw_actions" : raw_actions[timestep] 
                    }, step=episode * len(inside_temp_levels) + timestep) 
                             
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "ValR":f"{avg_val_reward:.1f}",
                        "Pwr":f"{val_power_mean:.1f}", 
                        "CO2": f"{val_co2_violation_mean:.1f}",
                        "Temp": f"{val_temp_violation_mean:.1f}"
                    }
                )
# Create eval experiment save dir



eval_season  = "mixed"
train_season = "mixed"
eval_env_id = "A403large"
train_env_id = "A403medium"
ENV_NAME = f"EVAL_{eval_env_id}"
ALGORITHM_NAME = "DQN"
current_date = datetime.now().strftime("%Y-%m-%d_%H:%M")
unique_experiment_name = f"eval_{eval_season}_{eval_env_id}_train_{train_season}_{train_env_id}_{current_date}"
experiment_save_dir_name = "results/dqn/" + unique_experiment_name
if not os.path.exists(experiment_save_dir_name):
    os.makedirs(experiment_save_dir_name)

#Replace with experiment path that is used to train.
model_load_path = "results/dqn/mixed_A403medium_train_2025-04-10_01:42"

config_file_path = os.path.join(model_load_path, "parameters_config.json")
# Ensure the file exists
if not os.path.exists(config_file_path):
    raise FileNotFoundError(f"Training config file not found at {config_file_path}")

# Load the training parameters from the JSON file
with open(config_file_path, "r") as f:
    training_params = json.load(f)

name= create_experiment_name(env_name=ENV_NAME, episodes=training_params['num_episodes']['value'],algorithm_name=ALGORITHM_NAME)

   
parameters_dict = {
    'learning_rate': training_params['learning_rate'],
    'lambda_energy': training_params['lambda_energy'],
    'energy_weight': training_params['energy_weight'],
    'co2_weight': training_params['co2_weight'],
    'temp_weight': training_params['temp_weight'],
    # Add evaluation-specific parameters:
    'model_load_path': {'value': model_load_path},
    'experiment_save_dir': {'value': experiment_save_dir_name},
    'eval_season': {'value': eval_season},
    'agent_count':  training_params['agent_count'],
    'num_episodes': training_params['num_episodes'],
    'env_id': {
        'value': eval_env_id
    }
}

sweep_config = {
    'method': 'grid',
    'name': name,
    'parameters': parameters_dict,
    'metric': {
        'name': 'val_total_power_kwh_mean',
        'goal': 'minimize'
    }
}


sweep_id = wandb.sweep(sweep_config, project="A403-Eval",entity="mehmetbh")
wandb.agent(sweep_id, evaluate, count=parameters_dict['agent_count']['value'])

#Close the agent

wandb.finish()