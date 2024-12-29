import math
class SetpointController():
    '''
    Setpoint controller for CO2 Fan is a hystreresis controller that turns on the fan on full speed when the CO2 concentration
    is above 700 ppm and turns off the fan when the CO2 concentration is below 600 ppm.
    '''
    # mapping = {

    def __init__(self):
        self.is_co2_open = False
        self.is_hvac_open = False
        self.summer_hvac_on_co2_off = 16
        self.summer_hvac_on_co2_on = 18
        self.winter_hvac_on_co2_off = 28
        self.winter_hvac_on_co2_on = 30
        self.off_action = 36
        self.summer_limits = [23,26]
        self.winter_limits = [20,23.5]

    def select_action(self, state):
        '''
        Act method for the controller
        '''
        co2 = state[0][-2]
        temp = state[0][9]
        
        month_sin = state[0][0]
        month_cos = state[0][1]
        month = int((math.atan2(month_sin, month_cos) * 12 / (2 * math.pi)) % 12) + 1

        # Determine season (summer or winter)
        is_summer = 6 <= month <= 9

        # Handle CO2 Fan hysteresis
        if co2 > 700:
            self.is_co2_open = True
        elif co2 < 600:
            self.is_co2_open = False

        # Summer HVAC control (cooling)
        if is_summer:
            if temp > self.summer_limits[1]:  # Above upper limit
                self.is_hvac_open = True
            elif temp < self.summer_limits[0]:  # Below lower limit
                self.is_hvac_open = False
            if self.is_hvac_open:
                return self.summer_hvac_on_co2_on if self.is_co2_open else self.summer_hvac_on_co2_off

        # Winter HVAC control (heating)
        else:
            if temp < self.winter_limits[0]:  # Below lower limit
                self.is_hvac_open = True
            elif temp > self.winter_limits[1]:  # Above upper limit
                self.is_hvac_open = False
            if self.is_hvac_open:
                return self.winter_hvac_on_co2_on if self.is_co2_open else self.winter_hvac_on_co2_off

        # If no conditions are met, turn off
        return self.off_action