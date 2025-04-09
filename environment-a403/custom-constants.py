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
# -------------------------- ACTION MAPPINGS -------------------------- #
A403_MAPPINGS = {
    "NO SPEED CONTROL": {
        0 : [20,22,1.0,0.0],
        1 : [20,22,1.0,1.0],
        2 : [22,24,1.0,0.0],
        3 : [22,24,1.0,1.0],
        4 : [24,26,1.0,0.0],
        5 : [24,26,1.0,1.0],
        6 : [5,50,0.0,0.0],
        7 : [5,50,0.0,1.0],
    },
    "FULL RESOULUTION CONTROL": {
        0 : [20, 21, 0.5, 0.0],
        1 : [20, 21, 0.5, 0.5],
        2 : [20, 21, 0.5, 0.75],
        3 : [20, 21, 0.5, 1.0],
        4 : [20, 21, 0.75, 0.0],
        5 : [20, 21, 0.75, 0.5],
        6 : [20, 21, 0.75, 0.75],
        7 : [20, 21, 0.75, 1.0],
        8 : [20, 21, 1.0, 0.0],
        9 : [20, 21, 1.0, 0.5],
        10 : [20, 21, 1.0, 0.75],
        11 : [20, 21, 1.0, 1.0],
        12 : [21, 22, 0.5, 0.0],
        13 : [21, 22, 0.5, 0.5],
        14 : [21, 22, 0.5, 0.75],
        15 : [21, 22, 0.5, 1.0],
        16 : [21, 22, 0.75, 0.0],
        17 : [21, 22, 0.75, 0.5],
        18 : [21, 22, 0.75, 0.75],
        19 : [21, 22, 0.75, 1.0],
        20 : [21, 22, 1.0, 0.0],
        21 : [21, 22, 1.0, 0.5],
        22 : [21, 22, 1.0, 0.75],
        23 : [21, 22, 1.0, 1.0],
        24 : [22, 23, 0.5, 0.0],
        25 : [22, 23, 0.5, 0.5],
        26 : [22, 23, 0.5, 0.75],
        27 : [22, 23, 0.5, 1.0],
        28 : [22, 23, 0.75, 0.0],
        29 : [22, 23, 0.75, 0.5],
        30 : [22, 23, 0.75, 0.75],
        31 : [22, 23, 0.75, 1.0],
        32 : [22, 23, 1.0, 0.0],
        33 : [22, 23, 1.0, 0.5],
        34 : [22, 23, 1.0, 0.75],
        35 : [22, 23, 1.0, 1.0],
        36 : [23, 24, 0.5, 0.0],
        37 : [23, 24, 0.5, 0.5],
        38 : [23, 24, 0.5, 0.75],
        39 : [23, 24, 0.5, 1.0],
        40 : [23, 24, 0.75, 0.0],
        41 : [23, 24, 0.75, 0.5],
        42 : [23, 24, 0.75, 0.75],
        43 : [23, 24, 0.75, 1.0],
        44 : [23, 24, 1.0, 0.0],
        45 : [23, 24, 1.0, 0.5],
        46 : [23, 24, 1.0, 0.75],
        47 : [23, 24, 1.0, 1.0],
        48 : [24, 25, 0.5, 0.0],
        49 : [24, 25, 0.5, 0.5],
        50 : [24, 25, 0.5, 0.75],
        51 : [24, 25, 0.5, 1.0],
        52 : [24, 25, 0.75, 0.0],
        53 : [24, 25, 0.75, 0.5],
        54 : [24, 25, 0.75, 0.75],
        55 : [24, 25, 0.75, 1.0],
        56 : [24, 25, 1.0, 0.0],
        57 : [24, 25, 1.0, 0.5],
        58 : [24, 25, 1.0, 0.75],
        59:  [24, 25, 1.0, 1.0],
        60 : [25, 26, 0.5, 0.0],
        61 : [25, 26, 0.5, 0.5],
        62 : [25, 26, 0.5, 0.75],
        63 : [25, 26, 0.5, 1.0],
        64 : [25, 26, 0.75, 0.0],
        65 : [25, 26, 0.75, 0.5],
        66 : [25, 26, 0.75, 0.75],
        67 : [25, 26, 0.75, 1.0],
        68 : [25, 26, 1.0, 0.0],
        69 : [25, 26, 1.0, 0.5],
        70 : [25, 26, 1.0, 0.75],
        71 : [25, 26, 1.0, 1.0], 
        72: [5,50,0.0,0.0], # OFF ACTION FOR HVAC AND WINDOW FAN
        73: [5,50,0.0,0.5],
        74: [5,50,0.0,0.75],
        75: [5,50,0.0,1.0]
    },
    "ONLY HVAC SPEED": {
        0 : [19,21,0.5,0.0],
        1 : [19,21,0.5,1.0],
        2 : [19,21,0.75,0.0],
        3 : [19,21,0.75,1.0],
        4 : [19,21,1.0,0.0],
        5 : [19,21,1.0,1.0],
        6 : [21,23,0.5,0.0],
        7 : [21,23,0.5,1.0],
        8 : [21,23,0.75,0.0],
        9 : [21,23,0.75,1.0],
        10 : [21,23,1.0,0.0],
        11 : [21,23,1.0,1.0],
        12 : [23,26,0.5,0.0],
        13 : [23,26,0.5,1.0],
        14 : [23,26,0.75,0.0],
        15 : [23,26,0.75,1.0],
        16 : [23,26,1.0,0.0],
        17 : [23,26,1.0,1.0],
        18 : [5,50,0.0,0.0],
        19 : [5,50,0.0,1.0]
    },
    "ONLY FAN":{
            0: [ 5, 50, 0.0, 0.0],
            1: [ 5, 50, 0.0, 0.25],
            2: [ 5, 50, 0.0, 0.5],
            3: [ 5, 50, 0.0, 0.75],
            4: [ 5, 50, 0.0, 1.0],
    }
}
# -------------------------- MAPPING FUNCTION -------------------------- #
def get_a403_action_mapping(env_type: str, action) -> List[float]:
    """Generic mapper to resolve action index to control values."""
    if isinstance(action, np.ndarray):
        action = int(action.item())

    mapping = A403_MAPPINGS.get(env_type)
    if mapping is None:
        raise ValueError(f"Unknown environment type: {env_type}")

    if action not in mapping:
        raise IndexError(f"Invalid action {action} for mapping {env_type}")

    return mapping[action]


# Define the ranges for heating, cooling, and fan speeds
HEATING_RANGE = [16,18,20,21]  
COOLING_RANGE = [30,28,26,24,23.25]  
FAN_SPEEDS = [0.5,0.75,1.0]  # Fan speeds: [0.5, 0.75, 1.0]
WINDOW_FAN = [0.0,0.50,0.75,1.0] 

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
def DEFAULT_A403NEW_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping("ONLY HVAC SPEED", action)
def DEFAULT_A403V3_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping("ONLY HVAC SPEED", action)
def DEFAULT_A403_DISCRETE_FUNCTION(action: int) -> List[float]:
    return get_a403_action_mapping("ONLY HVAC SPEED", action)
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