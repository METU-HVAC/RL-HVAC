import os
import math
from datetime import datetime
import numpy as np
import torch
obs_means =  [-1.88616163e-01 ,-2.57461701e-01 , 2.59861111e-01, -2.36515783e-05,
 -1.79738013e-04  ,2.37658727e+01  ,3.70050000e+01,  2.21512500e+01,
  2.51512500e+01 , 2.34620032e+01 , 3.42801016e+01 , 1.71827337e+00,
  7.50636800e+02 , 2.44119660e-01  ,1.02112024e+01 , 4.52476931e+02,
  4.37303756e+05]
obs_stds =  [5.76098289e-01 ,7.52494634e-01, 4.38558222e-01, 7.07073908e-01,
 7.07139627e-01, 8.30790585e+00 ,2.19055985e+01 ,1.45889459e+00,
 1.45889459e+00 ,1.74003051e+00 ,1.61345113e+01 ,1.99147147e+00,
 6.55276029e+02 ,4.34552292e-01 ,5.27910243e+00 ,6.01012338e+01,
 3.89152352e+05]
def normalize_observation(observation, means, stds):
    """
    Normalize a raw observation using the provided means and standard deviations.
    
    Parameters:
        observation (array-like): Raw observation to normalize.
        means (array-like): List or array of mean values for each feature.
        stds (array-like): List or array of standard deviation values for each feature.
        
    Returns:
        np.ndarray: Normalized observation where each feature is scaled as (x - mean) / std.
    """
    if isinstance(observation, torch.Tensor):
        # Move to CPU if needed and convert to NumPy
        observation = observation.cpu().numpy()
    observation = np.array(observation)  # Ensure observation is a numpy array
    means = np.array(means)
    stds = np.array(stds)
    
    # Prevent division by zero for features with zero standard deviation
    stds[stds == 0] = 1.0
    
    normalized_obs = (observation - means) / stds
    return normalized_obs

def create_experiment_name(env_name, episodes,algorithm_name):
    experiment_date = datetime.today().strftime('%Y-%m-%d_%H:%M')
    experiment_name = algorithm_name+'-' + env_name + \
        '-episodes-' + str(episodes)
    experiment_name += '_' + experiment_date
    return experiment_name

def save_run_metrics(save_dir,avg_Train_reward, avg_Train_power, avg_total_co2_concentration,avg_occupnacy_co2_concentration):
    with open(os.path.join(save_dir, "run_metrics.txt"), "a") as f:
        f.write(f"Train Average Reward = {avg_Train_reward}\n")
        f.write(f"Train Average Power = {avg_Train_power}\n")
        f.write(f"Train Average Total CO2 Concentration = {avg_total_co2_concentration}\n")
        f.write(f"Train Average Occupancy CO2 Concentration = {avg_occupnacy_co2_concentration}\n")

def calculate_time_label(month_sin, month_cos, hour_sin, hour_cos, current_step, timesteps_per_hour):
    """
    Calculate the human-readable time label from sinusoidal time features and step information.

    Args:
        month_sin (float): Sine of the month.
        month_cos (float): Cosine of the month.
        hour_sin (float): Sine of the hour.
        hour_cos (float): Cosine of the hour.
        is_weekend (int): 1 if weekend, 0 otherwise.
        current_step (int): Current timestep index.
        timesteps_per_hour (int): Number of timesteps per hour.

    Returns:
        str: Formatted time label.
    """
    # Reconstruct the month (1-12)
    month_angle = math.atan2(month_sin, month_cos)
    month = int(((month_angle + 2 * math.pi) % (2 * math.pi)) * (12 / (2 * math.pi)) + 1)

    # Reconstruct the hour (0-23)
    hour_angle = math.atan2(hour_sin, hour_cos)
    hour = int(((hour_angle + 2 * math.pi) % (2 * math.pi)) * (24 / (2 * math.pi)))

    # Calculate the minute
    minute = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))

    time_label = f"Month: {month:02}, Hour: {hour:02}:{minute:02}"
    return time_label