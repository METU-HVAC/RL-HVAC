import gymnasium as gym
from datetime import datetime, timedelta


DEFAULT_CONFIG = {
    'temperature_variables': ['air_temperature'],
    'energy_variables': ['HVAC_electricity_demand_rate'],
    'range_comfort_winter': (20.0, 23.5),
    'range_comfort_summer': (23.0, 26.0),
    'energy_weight':  0.3,
    'lambda_energy':  1e-4,
    'lambda_temperature':1.0,
}

# Helper to create a new environment with given start and end dates
def create_environment(start_date, end_date, reward_fn,env_name='Eplus-A403-hot-discrete-v1', timesteps_per_hour=12, reward_kwargs=DEFAULT_CONFIG):
    """
    Helper function to create a Gymnasium environment with specific configurations.

    Args:
        
        start_date (datetime): Start date for the environment simulation.
        end_date (datetime): End date for the environment simulation.
        reward_fn (class): Custom reward class for the environment.
        env_name (str): The name of the environment to create.
        timesteps_per_hour (int): Number of timesteps per hour (default: 12 for 5-minute intervals).
        reward_kwargs (dict, optional): Additional parameters for the reward function.

    Returns:
        gym.Env: Configured Gymnasium environment instance.
    """
    extra_params = {
        'timesteps_per_hour': timesteps_per_hour,
        'runperiod': (
            start_date.day, start_date.month, start_date.year,
            end_date.day, end_date.month, end_date.year
        )
    }

    env = gym.make(env_name, 
                   reward=reward_fn,
                   reward_kwargs=reward_kwargs,
                   config_params=extra_params)
    return env