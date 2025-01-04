import os
import math
from datetime import datetime



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