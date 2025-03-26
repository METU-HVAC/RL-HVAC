import numpy as np

def calculate_co2_rew(desired,observed,scaling_factor):
    """
    Calculate the reward for the CO2 level.
    
    Args:
        desired (float): Desired CO2 level.
        observed (float): Observed CO2 level.
        
    Returns:
        float: Reward value.
    """
    # Calculate the absolute difference between desired and observed CO2 levels
    diff = np.abs(desired - observed)
    diff_sq = diff**2

    rew  = -5*(1 - np.exp(-diff_sq/scaling_factor))
    
    return rew

# Test the function over some values 
desired = 400

observed = np.linspace(400,2000,100)
scaling_factor = 1000000
rewards = [calculate_co2_rew(desired,obs,scaling_factor) for obs in observed]

import matplotlib.pyplot as plt     
plt.plot(observed,rewards)
plt.xlabel('Observed CO2 Level')
plt.ylabel('Reward')

plt.show()
