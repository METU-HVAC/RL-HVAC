import numpy as np
from sinergym.utils.constants import *

class RBCAgent:
    def __init__(self, heating_setpoint=21.0, cooling_setpoint=23.0, start_hour=9, end_hour=17,ac_fan =1.0,window_fan=0.0):
        """
        Initialize the Rule-Based Controller (RBC).

        Parameters:
        - heating_setpoint (float): Heating setpoint temperature in °C.
        - cooling_setpoint (float): Cooling setpoint temperature in °C.
        - start_hour (int): Hour to start controlling (inclusive).
        - end_hour (int): Hour to stop controlling (exclusive).
        - ac_fan (float): Air conditioning fan speed (0.0 to 1.0).
        - window_fan (float): Window fan speed (0.0 to 1.0).
        """
        self.heating_setpoint = heating_setpoint
        self.cooling_setpoint = cooling_setpoint
        self.start_hour = start_hour
        self.end_hour = end_hour
        self.ac_fan = ac_fan
        self.window_fan = window_fan

    def select_action(self, state):
        """
        Determine the setpoints based on the current time of day.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - np.ndarray: Action containing heating and cooling setpoints.
        """
        # Extract the hour from the state (assuming it is the 3rd index)
        hour = state[0][2]
        
        # Convert fractional hour to integer hour
        current_hour = int(hour)
        
        if self.start_hour <= current_hour < self.end_hour:
            # During working hours, set HVAC to active mode
            
            action = 14
        else:
            # Outside working hours, deactivate HVAC by setting wide bounds
            action = 15  
            #How to convert from ACTION MAPPING to single integer
            #action = 10
            
            
        return action