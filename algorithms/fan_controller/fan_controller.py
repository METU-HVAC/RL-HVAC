all_action_map = {
    0 : [21, 22, 1.0, 0.0],
    1 : [21, 22, 1.0, 0.5],
    2 : [21, 22, 1.0, 0.75],
    3 : [21, 22, 1.0, 1.0],
    4 : [22, 23, 1.0, 0.0],
    5 : [22, 23, 1.0, 0.5],
    6 : [22, 23, 1.0, 0.75],
    7 : [22, 23, 1.0, 1.0],
    8 : [23, 24, 1.0, 0.0],
    9 : [23, 24, 1.0, 0.5],
    10 : [23, 24, 1.0, 0.75],
    11 : [23, 24, 1.0, 1.0],
    12 : [24, 25, 1.0, 0.0],
    13 : [24, 25, 1.0, 0.5],
    14 : [24, 25, 1.0, 0.75],
    15 : [24, 25, 1.0, 1.0],
    16 : [25, 26, 1.0, 0.0],
    17 : [25, 26, 1.0, 0.5],
    18 : [25, 26, 1.0, 0.75],
    19 : [25, 26, 1.0, 1.0],
    20 : [26, 27, 1.0, 0.0],
    21 : [26, 27, 1.0, 0.5],
    22 : [26, 27, 1.0, 0.75],
    23 : [26, 27, 1.0, 1.0],
    24 : [27, 28, 1.0, 0.0],
    25 : [27, 28, 1.0, 0.5],
    26 : [27, 28, 1.0, 0.75],
    27 : [27, 28, 1.0, 1.0],
    28 : [28, 29, 1.0, 0.0],
    29 : [28, 29, 1.0, 0.5],
    30 : [28, 29, 1.0, 0.75],
    31 : [28, 29, 1.0, 1.0],
    32 : [29, 30, 1.0, 0.0],
    33 : [29, 30, 1.0, 0.5],
    34 : [29, 30, 1.0, 0.75],
    35 : [29, 30, 1.0, 1.0],
    36 : [5 , 50, 0.0, 0.0],
    37 : [5 , 50, 0.0, 0.5],
    38 : [5 , 50, 0.0, 0.75],
    39 : [5 , 50, 0.0, 1.0]
}
class FanController:
    def __init__(self):
        self.ac_speed = 0.0
        self.heating_setpoint = 5.0
        self.cooling_setpoint = 50.0
    def select_action(self, state, is_summer, current_step, timesteps_per_hour):
        co2 = state[0][10]
        pmv = state[0][12]
        occupancy = state[0][9]
        
        if occupancy == 0:
            # No one is present, turn off HVAC and fan
            return self.find_action_id(5.0, 50.0, 0.0, 0.0)

        # Decide fan speed based on CO2
        if co2 > 800:
            window_fan_speed = 1.0
        elif 750 <= co2 <= 800:
            window_fan_speed = 0.75
        elif 700 <= co2 < 750:
            window_fan_speed = 0.5
        else:
            window_fan_speed = 0.0

        # Construct action
        return self.find_action_id(self.heating_setpoint, self.cooling_setpoint, self.ac_speed, window_fan_speed)

    def find_action_id(self, heating, cooling, ac_speed, fan_speed):
        for action_id, values in all_action_map.items():
            if (values[0] == heating and
                values[1] == cooling and
                values[2] == ac_speed and
                values[3] == fan_speed):
                return action_id
        print(f"⚠️ No matching action found for: [{cooling}, {heating}, {ac_speed}, {fan_speed}]")
        return 0  # fallback

class HysteresisFanController:
    def __init__(self, set_fan_speed=0.5):
        self.ac_speed = 0.0
        self.heating_setpoint = 5.0
        self.cooling_setpoint = 50.0
        self.set_window_fan_speed = set_fan_speed
        self.window_fan_speed = 0.0
    def select_action(self, state, is_summer, current_step, timesteps_per_hour):
        co2 = state[0][10]
        pmv = state[0][12]
        occupancy = state[0][9]
        
        if occupancy == 0:
            # No one is present, turn off HVAC and fan
            return self.find_action_id(5.0, 50.0, 0.0, 0.0)

        # Decide fan speed based on CO2
        if co2 > 800:
            self.window_fan_speed = self.set_window_fan_speed
        elif co2 < 700:
            self.window_fan_speed = 0.0
        else:
            # Do not change
            pass

        # Construct action
        return self.find_action_id(self.heating_setpoint, self.cooling_setpoint, self.ac_speed, self.window_fan_speed)

    def find_action_id(self, heating, cooling, ac_speed, fan_speed):
        for action_id, values in all_action_map.items():
            if (values[0] == heating and
                values[1] == cooling and
                values[2] == ac_speed and
                values[3] == fan_speed):
                return action_id
        print(f"⚠️ No matching action found for: [{cooling}, {heating}, {ac_speed}, {fan_speed}]")
        return 0  # fallback
    
class OnOffFanController:
    def __init__(self, set_fan_speed=0.5):
        self.ac_speed = 0.0
        self.heating_setpoint = 5.0
        self.cooling_setpoint = 50.0
        self.set_window_fan_speed = set_fan_speed
        self.window_fan_speed = 0.0
    def select_action(self, state, is_summer, current_step, timesteps_per_hour):
        co2 = state[0][10]
        pmv = state[0][12]
        occupancy = state[0][9]
        
        if occupancy == 0:
            # No one is present, turn off HVAC and fan
            return self.find_action_id(5.0, 50.0, 0.0, 0.0)
        else:
            self.window_fan_speed = self.set_window_fan_speed

        # Construct action
        return self.find_action_id(self.heating_setpoint, self.cooling_setpoint, self.ac_speed, self.window_fan_speed)

    def find_action_id(self, heating, cooling, ac_speed, fan_speed):
        for action_id, values in all_action_map.items():
            if (values[0] == heating and
                values[1] == cooling and
                values[2] == ac_speed and
                values[3] == fan_speed):
                return action_id
        print(f"⚠️ No matching action found for: [{cooling}, {heating}, {ac_speed}, {fan_speed}]")
        return 0  # fallback