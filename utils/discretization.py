import numpy as np 
def discretize_action(action, low,high,num_value):

    # Define fixed bins in the range [-1, 1]
    bins = np.linspace(-1, 1, num_value + 1)
    
    # Define representative values for each bin
    values = np.linspace(low, high, num_value)  # Change 18.0 and 30.0 if bounds differ
    
    # Find the bin index for the action
    bin_index = np.digitize([action], bins) - 1  # Adjust for 0-based indexing
    
    # Clamp bin index to valid range
    bin_index = np.clip(bin_index[0], 0, num_value - 1)
    
    # Return the value corresponding to the bin
    return values[bin_index]

def discretize_actions(actions, lows, highs, num_values):
    """
    Discretizes a list of actions and applies logic for heating/cooling setpoints and fan speeds.

    Parameters:
    - actions: List of continuous actions (e.g., [-1, 0.5, 0.8, -0.3]).
    - lows: List of lower bounds for each action.
    - highs: List of upper bounds for each action.
    - num_values: List of the number of discrete values for each action.

    Returns:
    - List of discretized actions.
    """
    # Discretize all actions
    all_actions = [
        discretize_action(action, low, high, num_value)
        for action, low, high, num_value in zip(actions, lows, highs, num_values)
    ]
    
    # Initialize the discrete_action array
    discrete_action = all_actions[:]
    # Incmoing heating sp,hvac_fan,window_fan,open_or_close
    # output heating,cooling,hva,window_fan
    # Apply logic for HVAC and window fan behavior
    if all_actions[3] == 0:  # HVAC fan speed = 0
        discrete_action[0] = 5           # Heating setpoint (indicates HVAC off)
        discrete_action[1] = 50          # Cooling setpoint (HVAC off indicator)
        discrete_action[2] = 0           # HVAC fan speed
        discrete_action[3] = all_actions[2]              # Window fan speed
    else:
        discrete_action[0] = all_actions[0]              # Heating setpoint
        discrete_action[1] = all_actions[0] + 3          # Cooling setpoint
        discrete_action[2] = all_actions[1]              # HVAC fan speed
        discrete_action[3] = all_actions[2]              # Window fan speed
    return discrete_action

def scale_action(action, low, high):
    # Incoming values are between -1 and 1 and need to be mapped to low and high
    return low + (high - low) * (action + 1) / 2
    

def scale_actions(actions, lows, highs):
    # input Heating sp, hvac_fan, window_fan
    scaled_actions = [0,0,0,0]
    
    # output heating, cooling, hvac, window_fan
    scaled_actions[0] = scale_action(actions[0],lows[0],highs[0])
    scaled_actions[1] = scaled_actions[0] +44
    scaled_actions[2] = scale_action(actions[1],lows[1],highs[1])
    scaled_actions[3] = scale_action(actions[2],lows[2],highs[2])
    return scaled_actions