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
from experiments.future_prediction.predictor import FutureObservationProvider
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

timesteps_per_hour = 6

env = create_basic_environment(env_id='A403medium',
                                start_date=datetime(1997, 1, 1),
                                end_date=datetime(1997, 1, 1) + timedelta(days=364),
                                season='hot',
                                timesteps_per_hour=timesteps_per_hour
)

# Load predictor
csv_path = './logs/whole_year.csv'
n_future = 2
predictor = FutureObservationProvider(csv_path, n=n_future)

# Start simulation
state, info = env.reset()
done = False

step_count = 1

while not done and step_count < 5:
    #random action from 0 to 10
    action = random.randint(0, 10)
    next_state, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Convert state to dict
    obs_variables = env.get_wrapper_attr('observation_variables')
    state_dict = dict(zip(obs_variables, next_state))

    # Apply your time appending
    state_dict = append_info_and_time_to_dict(state_dict, info, step_count, timesteps_per_hour)
    current_time_label = state_dict['time_label']

    # Predict future
    future_obs = predictor.get_future_observations(current_time_label)

    # Extract current values
    current_temp = state_dict['outdoor_temperature']
    current_humidity = state_dict['outdoor_humidity']
    current_people = state_dict['people_occupant']

    # Build appended features
    appended_features = []
    for obs in future_obs:
        appended_features.append(obs['outdoor_temperature'])
        appended_features.append(obs['people_occupant'])

    # Debug print:
    print(f"\n--- Step {step_count} ---")
    print(f"Current  | Temp: {current_temp:.2f}  Humidity: {current_humidity:.2f}  People: {current_people:.1f}")

    for i, obs in enumerate(future_obs, 1):
        print(f"Future t+{i} | Temp: {obs['outdoor_temperature']:.2f}  Humidity: {obs['outdoor_humidity']:.2f}  People: {obs['people_occupant']:.1f}")

    # Augment state
    augmented_state = np.concatenate([next_state, np.array(appended_features, dtype=np.float32)])
    
    state = augmented_state
    step_count += 1

env.close()
