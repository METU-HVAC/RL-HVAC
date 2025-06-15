import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
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
SEED = 42

reward_config = {
            'temperature_variables': ['air_temperature'],
            'co2_variable': 'air_co2',
            'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'ac_energy_weight': 0.3,
            'fan_energy_weight': 0.3,
            'co2_weight': 0.3,
            'temperature_weight': 0.3,
            'lambda_energy': 1/1800000, # 1/100.000
            'lambda_temperature': 1.0,
            'lambda_co2': 1.0,
            'co2_threshold': 800,
        }

# Your customized environment creation function
def create_basic_environment(env_id, start_date, end_date, season, timesteps_per_hour):
    
    env = create_environment(env_id,start_date, end_date,season,CO2andTemperatureReward,episode_type="Validation",timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    return env
# Logging function
def run_random_simulation_and_log(env_id, start_date, end_date, season, timesteps_per_hour, log_csv_path):
    env = create_basic_environment(env_id, start_date, end_date, season, timesteps_per_hour)

    state, info = env.reset()
    current_step = 1
    # Define only the desired observation variables to log
    selected_obs = ['outdoor_temperature', 'outdoor_humidity', 'people_occupant']

    # Prepare CSV
    os.makedirs(os.path.dirname(log_csv_path), exist_ok=True)
    with open(log_csv_path, mode='w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['time'] + selected_obs)

        
        done = False
        while not done:
            # Random action
            action = 0

            # Environment step
            next_state, reward, terminated, truncated, info = env.step(action)
            next_obs_dict = dict(zip(env.get_wrapper_attr('observation_variables'), next_state))
            next_obs_dict = append_info_and_time_to_dict(next_obs_dict,info,current_step, timesteps_per_hour)
            done = terminated or truncated
            if done:
                break
            # Write data to CSV
            writer.writerow(
                [next_obs_dict['time_label']] + [next_obs_dict[var] for var in selected_obs]
            )
            current_step += 1
            state = next_state

        env.close()

# Example usage:
if __name__ == '__main__':
    env_id = 'A403medium'  # adapt your env_id
    start_date = datetime(1997, 1, 1)
    total_day = 364
    end_date = start_date + timedelta(days=total_day)
    season = 'hot'
    timesteps_per_hour = 6  # 10-minute resolution

    log_path = './logs/whole_year.csv'
    run_random_simulation_and_log(env_id, start_date, end_date, season, timesteps_per_hour, log_path)