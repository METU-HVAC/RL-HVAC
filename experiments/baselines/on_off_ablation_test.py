import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
from algorithms.dqn.dqn import *
from algorithms.onoff.on_off_controller import *
from algorithms.setpoint.setpoint_controller import *
from environments.reward import *
from environments.environment import CO2_AND_TEMP_REWARD_CONFIG
import torch
from sinergym.utils.wrappers import DatetimeWrapper
from common.utils import *
from environments.environment import create_environment
from utils.dataset import generate_chunks, split_chunks
from utils.visualization import plot_and_save, plot_csv_data
from utils.experiment_utils import *
from tqdm import tqdm
import wandb
import pandas as pd
## OBSERVATION SPACE 
# {'month': np.float32(7.0), 'day_of_month': np.float32(10.0), 'hour': np.float32(0.0),
#  'outdoor_temperature': np.float32(28.666666), 'outdoor_humidity': np.float32(36.666668), '
# htg_setpoint': np.float32(4.13), 'clg_setpoint': np.float32(50.0), 'air_temperature': np.float32(26.72595), 
# 'air_humidity': np.float32(40.54236), 'people_occupant': np.float32(0.0), 'air_co2': np.float32(456.72827), 
# 'window_fan_energy': np.float32(0.0), 'total_electricity_HVAC': np.float32(0.0)}
ENV_NAME = "A403_V3"
ALGORITHM_NAME = "ON_OFF_ABL"
NUM_EPISODES = 1

raw_observations = []
log_val_dict = []
def run_simulation(start_date, end_date, season,episode_type, steps_per_chunk,agent,train_interval,timesteps_per_hour,reward_config):
    env = create_environment(start_date, end_date,season,CO2andTemperatureReward,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    
    state, info = env.reset()
    OFF_ACTION = 0 #initially the the system is not working
    state = append_fan_speed_to_observation(env,state,OFF_ACTION)

    data = state
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False

    current_step = 1 
    total_reward = 0
    loss_list = []
    all_obs_dict = {}
    while current_step < steps_per_chunk:

        action = agent.select_action(state)
        

        #np_action = np.array([action], dtype=np.float32)  # Adjust dtype to match environment

        observation, reward, truncated, terminated, info = env.step(action)
        observation = append_fan_speed_to_observation(env,observation,action)

        raw_observations.append(observation)

        #observation, reward, truncated, terminated, info = env.step(action.item())
        data = observation
        done = terminated or truncated
        reward = torch.tensor([reward],dtype=torch.float32, device=device)
        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        state = next_state

        obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), observation))

        obs_dict = append_info_and_time_to_dict(obs_dict,info,current_step, timesteps_per_hour)
       
        obs_dict = append_fan_speed_to_dict(obs_dict, DEFAULT_A403V3_DISCRETE_FUNCTION(action)[3], DEFAULT_A403V3_DISCRETE_FUNCTION(action)[2])
        all_obs_dict = add_observation(all_obs_dict, obs_dict)
        
        total_reward += reward
        
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
        
        seed = 42  # Set seed for reproducibility
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        remove_previous_run_logs()
                
        state_size =  17 # Adjust based on the size of your observation space
        action_size = 37  
        train_interval = 100 # Train every n steps
        timesteps_per_hour = 6  # 10-minute intervals
        days_per_chunk = 10
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        num_episodes = 10  # Total episodes for training
        start_date = datetime(1997, 1, 1)
        days_per_chunk = 10
        total_days = 365
        plots_dir = "results/plots/setpoint"  # Directory to store plots

        extra_params = {
            'timesteps_per_hour': timesteps_per_hour,
            'runperiod':(1,1,1997,12,3,1997)  # Full year simulation
        }

        # Generate and split chunks
        chunks = generate_chunks(start_date, days_per_chunk, total_days,seasons=["hot"])
        train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.8, val_ratio=0.2, seed=seed)

        num_episodes = NUM_EPISODES  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)
        total_number_of_test_chunks = len(test_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        total_testing_steps = total_number_of_test_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        reward_config = {
            'temperature_variables': ['air_temperature'],
            'co2_variable': 'air_co2',
            'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'energy_weight': 0.3,
            'co2_weight': 0.3,
            'temperature_weight': 0.3,
            'lambda_energy': 1e-5,
            'lambda_temperature': 1.0,
            'lambda_co2': 1.0,
            'co2_threshold': 800,
        }
        # Initialize the SP agent
        agent = OnOffController()
        # Create the main experiment directory (only once)
        experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(plots_dir, f"experiment_{experiment_timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
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
                train_temp_viol_percentage_list = []
                total_loss_list = []
                train_obs_dict = {}
                for train_chunk in train_chunks:
                    obs_dict = {} 
                    reward,loss,obs_dict = run_simulation(*train_chunk,
                                                                            "Training", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                    
                    total_loss_list.append(loss)
                    train_obs_dict = update_combined_dict(obs_dict, train_obs_dict)
                    train_total_reward += reward
                    #KPI's
                    window_power = sum(obs_dict['window_fan_energies'])
                    hvac_power = sum(obs_dict['total_electricity_HVACs'])
                    total_power = window_power + hvac_power
                    temp_viol_percentage = sum(obs_dict['temp_violations'])/len(obs_dict['temp_violations'])*100
                    co2_viol_percentage = sum(obs_dict['co2_violations'])/len(obs_dict['co2_violations'])*100

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
                    break # No need to train

                print(f"Loss for episode {episode}: {np.mean(total_loss_list)}")    
                avg_train_reward = (train_total_reward / len(train_chunks)).item()

                
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
                val_total_reward = 0
                val_total_power_list = []
                val_hvac_power_list = []
                val_fan_power_list = []
                val_co2_viol_percentage_list = []
                val_temp_viol_percentage_list = []
                val_obs_dict = {}
                for val_chunk in val_chunks:
                    obs_dict = {}    
                    reward ,loss,obs_dict = run_simulation(*val_chunk,
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

                    pbar.set_postfix_str(f"val Chunk {pbar.n + 1}/{len(val_chunks)}")
                    pbar.update(1)
                
                    
                #Save the dictionary to csv file
                save_observations_to_csv(log_val_dict, "on_off")    
                    
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
                #Temperature
                wandb.log({"train_temp_violation_mean":train_temp_violation_mean},step=episode * log_length)
                wandb.log({"train_temp_violation_std":train_temp_violation_std},step=episode * log_length)
                wandb.log({"val_temp_violation_mean":val_temp_violation_mean},step=episode * log_length)
                wandb.log({"val_temp_violation_std":val_temp_violation_std},step=episode * log_length)
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
                    }, step=episode * len(inside_temp_levels) + timestep) 
                             
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "TrR": f"{avg_train_reward:.1f}",
                        "ValR":f"{avg_val_reward:.1f}",
                        "AvgPwr":f"{train_power_mean:.1f}", 
                        "AvgCo2OccConc": f"{np.mean(val_co2_violation_mean):.1f}",
                    }
                )

name= create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)
sweep_config = {
    'method': 'grid' ,
    'name' : name
    }
metric = {
    'name': 'val_total_power_kwh_mean',
    'goal': 'minimize'   
    }
parameters_dict = ({
    'action': {
        'values': [1,2,3,4]
      }
    })
sweep_config['parameters'] = parameters_dict
sweep_config['metric'] = metric

sweep_id = wandb.sweep(sweep_config, project="A403-Train",entity="mehmetbh")
wandb.agent(sweep_id, train, count=1)

#Close the agent

wandb.finish()