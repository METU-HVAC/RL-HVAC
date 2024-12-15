"""Constants used in whole project."""

import os
from typing import List, Union

import numpy as np
import pkg_resources

# ---------------------------------------------------------------------------- #
#                               Generic constants                              #
# ---------------------------------------------------------------------------- #
# Sinergym Data path
PKG_DATA_PATH = pkg_resources.resource_filename(
    'sinergym', 'data/')
# Weekday encoding for simulations
WEEKDAY_ENCODING = {'monday': 0, 'tuesday': 1, 'wednesday': 2, 'thursday': 3,
                    'friday': 4, 'saturday': 5, 'sunday': 6}
# Default start year (Non leap year please)
YEAR = 1991
# cwd
CWD = os.getcwd()

# Logger values (environment layer, simulator layer and modeling layer)
# LOG_ENV_LEVEL = 'INFO'
# LOG_SIM_LEVEL = 'INFO'
# LOG_MODEL_LEVEL = 'INFO'
# LOG_WRAPPERS_LEVEL = 'INFO'
# LOG_REWARD_LEVEL = 'INFO'
# LOG_COMMON_LEVEL = 'INFO'
# LOG_CALLBACK_LEVEL = 'INFO'
LOG_ENV_LEVEL = 'WARNING'
LOG_SIM_LEVEL = 'WARNING'
LOG_MODEL_LEVEL = 'WARNING'
LOG_WRAPPERS_LEVEL = 'WARNING'
LOG_REWARD_LEVEL = 'WARNING'
LOG_COMMON_LEVEL = 'WARNING'
LOG_CALLBACK_LEVEL = 'WARNING'
# LOG_FORMAT = "[%(asctime)s] %(name)s %(levelname)s:%(message)s"
LOG_FORMAT = "[%(name)s] (%(levelname)s) : %(message)s"

# ---------------------------------------------------------------------------- #
#              Custom Eplus discrete environments action mappings             #
# ---------------------------------------------------------------------------- #

# --------------------------------------A403----------------------------------- #


# Define the ranges for heating, cooling, and fan speeds
HEATING_RANGE = np.linspace(16, 30, 5)  # 5 points between 16°C and 30 °C
COOLING_RANGE = np.linspace(18, 30, 5)  # 5 points between 18°C and 30°C
FAN_SPEEDS = np.linspace(0.25, 1.0, 4)  # Fan speeds: [0.0, 0.25, 0.5, 0.75, 1.0]
WINDOW_FAN = [0] # For the time being, it is not used.

# Pre-compute all combinations of [heating, cooling, fan speed]
ACTION_MAPPING = [
    [htg, clg, fan,window_fan]
    for htg in HEATING_RANGE
    for clg in COOLING_RANGE
    if htg < clg  # Ensure heating is lower than cooling
    for fan in FAN_SPEEDS
    for window_fan in WINDOW_FAN
]
# Add a dummy action for system "off" mode
OFF_ACTION = [5, 50, 0.0, 0.0]  # Heating at 5°C, Cooling at 50°C, Fan speed 0
ACTION_MAPPING.append(OFF_ACTION)
# ---------------------------------------------------------------------------- #
#              Default Eplus discrete environments action mappings             #
# ---------------------------------------------------------------------------- #

# -------------------------------------5ZONE---------------------------------- #

def DEFAULT_A403_DISCRETE_FUNCTION(action: int) -> List[float]:
    """Maps a discrete action to continuous [heating, cooling, fan speed]."""
    if isinstance(action, np.ndarray):
        action = int(action.item())  # Handle ndarray input

    #mapped_action = np.array(ACTION_MAPPING[action], dtype=np.float32)  # Convert to NumPy array
    # mapping = {
    #     0 : [16, 30, 0.5, 0.0],
    #     1 : [16, 30, 0.75, 0.0],
    #     2 : [16, 30, 1.0, 0.0],
    #     3 : [18, 28, 0.5, 0.0],
    #     4 : [18, 28, 0.75, 0.0],
    #     5 : [18, 28, 1.0, 0.0],
    #     6 : [20, 26, 0.5, 0.0],
    #     7 : [20, 26, 0.75, 0.0],
    #     8 : [20, 26, 1.0, 0.0],
    #     9 : [21, 24, 0.5, 0.0],
    #     10 : [21, 24, 0.75, 0.0],
    #     11 : [21, 24, 1.0, 0.0],
    #     12 : [21, 23.25, 0.5, 0.0],
    #     13 : [21, 23.25, 0.75, 0.0],
    #     14 : [21, 23.25, 1.0, 0.0],
    #     15 : OFF_ACTION
    # }
    
    #mapped_action = np.array(ACTION_MAPPING[action], dtype=np.float32)  # Convert to NumPy array
    #Only Fan
    mapping = {
        0: [ 5, 50, 0.0, 0.0],
        1: [ 5, 50, 0.0, 0.25],
        2: [ 5, 50, 0.0, 0.5],
        3: [ 5, 50, 0.0, 0.75],
        4: [ 5, 50, 0.0, 1.0],
        5 : OFF_ACTION
    }
    mapped_action = mapping[action]
    return mapped_action
def DEFAULT_5ZONE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [12, 30],
        1: [13, 30],
        2: [14, 29],
        3: [15, 28],
        4: [16, 28],
        5: [17, 27],
        6: [18, 26],
        7: [19, 25],
        8: [20, 24],
        9: [21, 23.25]
    }

    return mapping[action]


# ----------------------------------DATACENTER--------------------------------- #

def DEFAULT_DATACENTER_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------WAREHOUSE--------------------------------- #


def DEFAULT_WAREHOUSE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICE--------------------------------- #


def DEFAULT_OFFICE_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# ----------------------------------OFFICEGRID---------------------------- #


def DEFAULT_OFFICEGRID_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30, 0.0, 0.0],
        1: [16, 29, 0.0, 0.0],
        2: [17, 28, 0.0, 0.0],
        3: [18, 27, 0.0, 0.0],
        4: [19, 26, 0.0, 0.0],
        5: [20, 25, 0.0, 0.0],
        6: [21, 24, 0.0, 0.0],
        7: [22, 23, 0.0, 0.0],
        8: [22, 22.5, 0.0, 0.0],
        9: [21, 22.5, 0.0, 0.0]
    }

    return mapping[action]

# ----------------------------------SHOP--------------------- #


def DEFAULT_SHOP_DISCRETE_FUNCTION(action: int) -> List[float]:
    # SB3 algotihms returns a ndarray instead of a int
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = {
        0: [15, 30],
        1: [16, 29],
        2: [17, 28],
        3: [18, 27],
        4: [19, 26],
        5: [20, 25],
        6: [21, 24],
        7: [22, 23],
        8: [22, 22.5],
        9: [21, 22.5]
    }

    return mapping[action]

# -------------------------------- AUTOBALANCE ------------------------------- #


def DEFAULT_RADIANT_DISCRETE_FUNCTION(
        action: Union[np.ndarray, List[int]]) -> List[float]:
    action[5] += 25
    return list(action)


# ----------------------------------HOSPITAL--------------------------------- #
# DEFAULT_HOSPITAL_OBSERVATION_VARIABLES = [
#     'Zone Air Temperature(Basement)',
#     'Facility Total HVAC Electricity Demand Rate(Whole Building)',
#     'Site Outdoor Air Drybulb Temperature(Environment)'
# ]

# DEFAULT_HOSPITAL_ACTION_VARIABLES = [
#     'hospital-heating-rl',
#     'hospital-cooling-rl',
# ]

# DEFAULT_HOSPITAL_OBSERVATION_SPACE = gym.spaces.Box(
#     low=-5e6,
#     high=5e6,
#     shape=(len(DEFAULT_HOSPITAL_OBSERVATION_VARIABLES) + 4,),
#     dtype=np.float32)

# DEFAULT_HOSPITAL_ACTION_MAPPING = {
#     0: (15, 30),
#     1: (16, 29),
#     2: (17, 28),
#     3: (18, 27),
#     4: (19, 26),
#     5: (20, 25),
#     6: (21, 24),
#     7: (22, 23),
#     8: (22, 22),
#     9: (21, 21)
# }

# DEFAULT_HOSPITAL_ACTION_SPACE_DISCRETE = gym.spaces.Discrete(10)

# DEFAULT_HOSPITAL_ACTION_SPACE_CONTINUOUS = gym.spaces.Box(
#     low=np.array([15.0, 22.5], dtype=np.float32),
#     high=np.array([22.5, 30.0], dtype=np.float32),
#     shape=(2,),
#     dtype=np.float32)

# DEFAULT_HOSPITAL_ACTION_DEFINITION = {
#     '': {'name': '', 'initial_value': 21},
#     '': {'name': '', 'initial_value': 25}
# }
