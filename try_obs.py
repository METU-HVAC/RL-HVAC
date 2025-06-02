# Simulate test scenario for dictionary-based agent observation extraction

from typing import List
import numpy as np

# Sample observation variables in the correct order
observation_variables = [
    'month', 'day_of_month', 'hour',
    'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature',
    'air_humidity', 'people_occupant', 'air_co2',
    'window_fan_energy', 'total_electricity_HVAC'
]

# Corresponding observation values as would be received from the environment
observation_values = [
    7.0, 10.0, 31.0,              # time variables
    28.67, 36.67,                # outdoor temp & humidity
    4.13, 50.0,                  # setpoints
    26.73, 40.54, 0.0, 456.73,   # indoor temp/humidity, people, CO2
    60000.0, 700000.0                     # energy consumption
]

# Simulate a fan/AC action to pass to speed mapping
def DEFAULT_A403V3_DISCRETE_FUNCTION(action):
    # Return (dummy temp_min, temp_max, ac_speed, fan_speed)
    # Just for testing
    return (21, 24, 1.0, 0.5)

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
    obs_dict = dict(zip(observation_variables, observation))

    obs_dict = append_fan_speed_to_dict(
        obs_dict,
        DEFAULT_A403V3_DISCRETE_FUNCTION(action)[3],
        DEFAULT_A403V3_DISCRETE_FUNCTION(action)[2]
    )
    obs_dict['is_summer'] = is_summer(obs_dict['month'])

    if agent_name == "WindowFan":
        keys = ['hour', 'air_co2', 'window_fan_energy', 'people_occupant']
    elif agent_name == "HVAC":
        keys = ['hour','outdoor_temperature','air_temperature', 'people_occupant', 'window_fan_speed', 'is_summer', 'total_electricity_HVAC']
    else:
        raise ValueError(f"Unknown agent: {agent_name}")
    
    return [obs_dict[k] for k in keys]

# Test both agents
fan_obs = get_agent_observation_dict_based("WindowFan", observation_values, action=3)
hvac_obs = get_agent_observation_dict_based("HVAC", observation_values, action=3)

print("Window Fan Observation:", fan_obs)
print("HVAC Observation:", hvac_obs)
