import math
class SetpointController():
    '''
    Setpoint controller for CO2 Fan is a hystreresis controller that turns on the fan on full speed when the CO2 concentration
    is above 700 ppm and turns off the fan when the CO2 concentration is below 600 ppm.
    '''
    def __init__(self,window_fan_speed=0.5):
        self.is_co2_open = False
        self.is_hvac_open = False
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_05 = 21
        self.winter_hvac_on_co2_075 = 22
        self.winter_hvac_on_co2_1 = 23
        
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_05 = 33
        self.summer_hvac_on_co2_075 = 34
        self.summer_hvac_on_co2_1 = 35
        
        self.hvac_off_co2_off = 36
        self.hvac_off_co2_05 = 37 
        self.hvac_off_co2_075 = 38  
        self.hvac_off_co2_1 = 39
         
        self.summer_limits = [23,26]
        self.winter_limits = [20,23.5]
        
        self.window_fan_speed = window_fan_speed

    def select_action(self, state,current_step,timesteps_per_hour):
        '''
        Act method for the controller
        '''
        co2 = state[0][10]
        temp = state[0][7]

        month = state[0][0]
        
        occupancy = state[0][9]
        # Determine season (summer or winter)
        is_summer = 6 <= month <= 9
        # Select seasonal limits based on the current season
        lower_limit, upper_limit = (
            self.summer_limits if is_summer else self.winter_limits
        )
        
        # Add hysteresis margins
        lower_hysteresis = lower_limit - 0.5
        upper_hysteresis = upper_limit + 0.5
        # Determine HVAC state based on the current temperature
        if not self.is_hvac_open:
            # If HVAC is off, decide to turn it on
            if temp < lower_limit:  # Too cold, need heating
                self.is_hvac_open = True
                self.is_cooling = False  # Enter heating mode
            elif temp > upper_limit:  # Too hot, need cooling
                self.is_hvac_open = True
                self.is_cooling = True  # Enter cooling mode
        else:
            # If HVAC is already on, use hysteresis to decide when to turn it off
            if self.is_cooling and temp <= lower_hysteresis:  # Cooling complete
                self.is_hvac_open = False
            elif not self.is_cooling and temp >= upper_hysteresis:  # Heating complete
                self.is_hvac_open = False
        # Handle CO2 Fan hysteresis
        if co2 > 800:
            self.is_co2_open = True
        elif co2 < 700:
            self.is_co2_open = False

        if occupancy > 0:
            if self.is_co2_open:
                if self.window_fan_speed == 0.5:
                    return self.summer_hvac_on_co2_05 if is_summer else self.winter_hvac_on_co2_05
                elif self.window_fan_speed == 0.75:
                    return self.summer_hvac_on_co2_075 if is_summer else self.winter_hvac_on_co2_075
                elif self.window_fan_speed == 1.0:
                    return self.summer_hvac_on_co2_1 if is_summer else self.winter_hvac_on_co2_1
                else:
                    print("Invalid window fan speed")
            else:
                return self.summer_hvac_on_co2_off if is_summer else self.winter_hvac_on_co2_off
                
        else:
            #If not in working hours, close everything
            return self.hvac_off_co2_off

class MultiSpeedSetpointController():
    def __init__(self):
        self.is_co2_open = False
        self.is_hvac_open = False
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_05 = 21
        self.winter_hvac_on_co2_075 = 22
        self.winter_hvac_on_co2_1 = 23
        
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_05 = 33
        self.summer_hvac_on_co2_075 = 34
        self.summer_hvac_on_co2_1 = 35
        
        self.hvac_off_co2_off = 36
        self.hvac_off_co2_05 = 37 
        self.hvac_off_co2_075 = 38  
        self.hvac_off_co2_1 = 39
         
        self.summer_limits = [23,26]
        self.winter_limits = [20,23.5]
        

    def select_action(self, state,current_step,timesteps_per_hour):
        '''
        Act method for the controller
        '''
        co2 = state[0][10]
        temp = state[0][7]

        month = state[0][0]
        
        occupancy = state[0][9]
        # Determine season (summer or winter)
        is_summer = 6 <= month <= 9
        # CO2-based fan speed control
        if co2 > 800:
            self.window_fan_speed = 1.0  # Full speed
        elif 750 <= co2 <= 800:
            self.window_fan_speed = 0.75  # High speed
        elif 700 <= co2 < 750:
            self.window_fan_speed = 0.5  # Low speed
        else:
            self.window_fan_speed = 0.0  # Off
            
        if occupancy > 0:
            if self.window_fan_speed > 0:
                if self.window_fan_speed == 0.5:
                    return self.summer_hvac_on_co2_05 if is_summer else self.winter_hvac_on_co2_05
                elif self.window_fan_speed == 0.75:
                    return self.summer_hvac_on_co2_075 if is_summer else self.winter_hvac_on_co2_075
                elif self.window_fan_speed == 1.0:
                    return self.summer_hvac_on_co2_1 if is_summer else self.winter_hvac_on_co2_1
                else:
                    print("Invalid window fan speed")
            else:
                return self.summer_hvac_on_co2_off if is_summer else self.winter_hvac_on_co2_off
                
        else:
            #If not in working hours, close everything
            return self.hvac_off_co2_off

class SingleSpeedACOnlyController():
    '''
    Setpoint controller for CO2 Fan is a hystreresis controller that turns on the fan on full speed when the CO2 concentration
    is above 700 ppm and turns off the fan when the CO2 concentration is below 600 ppm.
    '''
    def __init__(self,window_fan_speed=0.0):
        self.is_co2_open = False
        self.is_hvac_open = False
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_05 = 21
        self.winter_hvac_on_co2_075 = 22
        self.winter_hvac_on_co2_1 = 23
        
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_05 = 33
        self.summer_hvac_on_co2_075 = 34
        self.summer_hvac_on_co2_1 = 35
        
        self.hvac_off_co2_off = 36
        self.hvac_off_co2_05 = 37 
        self.hvac_off_co2_075 = 38  
        self.hvac_off_co2_1 = 39
         
        self.summer_limits = [23,26]
        self.winter_limits = [20,23.5]
        
        self.window_fan_speed = window_fan_speed

    def select_action(self, state,current_step,timesteps_per_hour):
        '''
        Act method for the controller
        '''
        co2 = state[0][10]
        temp = state[0][7]

        month = state[0][0]
        
        occupancy = state[0][9]
        # Determine season (summer or winter)
        is_summer = 6 <= month <= 9
        # Select seasonal limits based on the current season
        lower_limit, upper_limit = (
            self.summer_limits if is_summer else self.winter_limits
        )
        
        # Add hysteresis margins
        lower_hysteresis = lower_limit - 0.5
        upper_hysteresis = upper_limit + 0.5
        # Determine HVAC state based on the current temperature
        if not self.is_hvac_open:
            # If HVAC is off, decide to turn it on
            if temp < lower_limit:  # Too cold, need heating
                self.is_hvac_open = True
                self.is_cooling = False  # Enter heating mode
            elif temp > upper_limit:  # Too hot, need cooling
                self.is_hvac_open = True
                self.is_cooling = True  # Enter cooling mode
        else:
            # If HVAC is already on, use hysteresis to decide when to turn it off
            if self.is_cooling and temp <= lower_hysteresis:  # Cooling complete
                self.is_hvac_open = False
            elif not self.is_cooling and temp >= upper_hysteresis:  # Heating complete
                self.is_hvac_open = False
        # Handle CO2 Fan hysteresis
        if co2 > 800:
            self.is_co2_open = True
        elif co2 < 700:
            self.is_co2_open = False

        if occupancy > 0:
            if self.is_co2_open:
                if self.window_fan_speed == 0.5:
                    return self.summer_hvac_on_co2_05 if is_summer else self.winter_hvac_on_co2_05
                elif self.window_fan_speed == 0.75:
                    return self.summer_hvac_on_co2_075 if is_summer else self.winter_hvac_on_co2_075
                elif self.window_fan_speed == 1.0:
                    return self.summer_hvac_on_co2_1 if is_summer else self.winter_hvac_on_co2_1
                else:
                    print("Invalid window fan speed")
            else:
                return self.summer_hvac_on_co2_off if is_summer else self.winter_hvac_on_co2_off
                
        else:
            #If not in working hours, close everything
            return self.hvac_off_co2_off