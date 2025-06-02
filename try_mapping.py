# Action dictionary
reduced_actions = {
    0: [21.0, 23.0, 1.0, 0.0],
    1: [21.0, 23.0, 1.0, 1.0],
    2: [23.0, 26.0, 1.0, 0.0],
    3: [23.0, 26.0, 1.0, 1.0],
    4: [5, 50, 0.0, 0.0],
    5: [5, 50, 0.0, 1.0],
}

fan_map = {
    0: 0.0,   # Off
    1: 0.25,  # Low
    2: 0.5,   # Medium
    3: 1.0    # High
}

hvac_map = {
    0: [5.0, 50.0, 0.0],     # Off
    1: [23.0, 26.0, 1.0],    # Summer
    2: [21.0, 23.0, 1.0]     # Winter
}

from itertools import product

def generate_combined_action_dict(fan_map: dict, hvac_map: dict) -> dict:
    combined_actions = {}
    idx = 0

    for hvac_action, fan_action in product(hvac_map.values(), fan_map.values()):
        t_min, t_max, hvac_flag = hvac_action
        fan_flag = fan_action
        combined_actions[idx] = [t_min, t_max, hvac_flag, fan_flag]
        idx += 1

    return combined_actions

def get_combined_action_key(fan_action: int, 
                            hvac_action: int, 
                            action_dict: dict, 
                            fan_map: dict, 
                            hvac_map: dict) -> int:
    """
    Returns the key from the action_dict that matches the combination of fan and hvac actions.

    :param fan_action: int, key from fan_map (e.g. 0 or 1)
    :param hvac_action: int, key from hvac_map (e.g. 0, 1, 2)
    :param action_dict: dict, e.g. {0: [21, 23, 1.0, 0.0], ...}
    :param fan_map: dict, e.g. {0: 0.0, 1: 1.0}
    :param hvac_map: dict, e.g. {0: [5, 50, 0.0], 1: [23, 26.0, 1.0], 2: [21, 23, 1.0]}
    :return: int, key from action_dict
    """
    if fan_action not in fan_map:
        raise ValueError(f"Invalid fan_action: {fan_action}")
    if hvac_action not in hvac_map:
        raise ValueError(f"Invalid hvac_action: {hvac_action}")

    # Extract action representation
    fan_flag = fan_map[fan_action]
    t_min, t_max, hvac_flag = hvac_map[hvac_action]

    target_action = [t_min, t_max, hvac_flag, fan_flag]

    # Search for matching action in dict
    for key, value in action_dict.items():
        if value == target_action:
            return key

    raise ValueError("Combined action not found in the provided dictionary.")

def combine_actions(fan_action, hvac_action, action_dict, fan_map, hvac_map):
    fan_speed = fan_map[fan_action]
    t_min, t_max, hvac_flag = hvac_map[hvac_action]
    combined = [t_min, t_max, hvac_flag, fan_speed]

    for key, value in action_dict.items():
        if value == combined:
            return key
    raise ValueError("Invalid fan+hvac action combination")


reduced_actions = generate_combined_action_dict(fan_map, hvac_map)

for key, value in reduced_actions.items():
    print(f"{key}: {value}")

fan_action = 0
hvac_action = 1
action_key = get_combined_action_key(fan_action, hvac_action, reduced_actions, fan_map, hvac_map)
print(f"Combined action key for fan {fan_action} and HVAC {hvac_action}: {action_key}")
