
import math

#         12 : [20, 23, 0.5, 0.0],
#         13 : [20, 23, 0.5, 0.5],
#         14 : [20, 23, 0.5, 0.75],
#         15 : [20, 23, 0.5, 1.0],
#         16 : [20, 23, 0.75, 0.0],
#         17 : [20, 23, 0.75, 0.5],
#         18 : [20, 23, 0.75, 0.75],
#         19 : [20, 23, 0.75, 1.0],
#         20 : [20, 23, 1.0, 0.0],
#         21 : [20, 23, 1.0, 0.5],
#         22 : [20, 23, 1.0, 0.75],
#         23 : [20, 23, 1.0, 1.0],
#         24 : [23, 26, 0.5, 0.0],
#         25 : [23, 26, 0.5, 0.5],
#         26 : [23, 26, 0.5, 0.75],
#         27 : [23, 26, 0.5, 1.0],
#         28 : [23, 26, 0.75, 0.0],
#         29 : [23, 26, 0.75, 0.5],
#         30 : [23, 26, 0.75, 0.75],
#         31 : [23, 26, 0.75, 1.0],
#         32 : [23, 26, 1.0, 0.0],
#         33 : [23, 26, 1.0, 0.5], 
#         34 : [23, 26, 1.0, 0.75],
#         35 : [23, 26, 1.0, 1.0],
#         36 : [5 , 50, 0.0, 0.0], # OFF ACTION FOR HVAC AND WINDOW FAN
#         37 : [5 , 50, 0.0, 0.5],
#         38 : [5 , 50, 0.0, 0.75],
#         39 : [5 , 50, 0.0, 1.0]
        
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    #     20 : [21, 23, 1.0, 0.0],winter hvac on co2 off
    #     21 : [21, 23, 1.0, 0.5],winter hvac on co2 on 0.5
    #     22 : [21, 23, 1.0, 0.75],winter hvac on co2 on 0.75
    #     23 : [21, 23, 1.0, 1.0], winter hvac on co2 on 1.0

    #     32 : [23, 26, 1.0, 0.0], summer hvac on co2 off
    #     33 : [23, 26, 1.0, 0.5],summer hvac on co2 on 0.5
    #     34 : [23, 26, 1.0, 0.75],summer hvac on co2 on 0.75
    #     35 : [23, 26, 1.0, 1.0],summer hvac on co2 on 1.0
    
    #     36 : [5 , 50, 0.0, 0.0], off action
    #     37 : [5 , 50, 0.0, 0.5], hvac off co2 0.5
    #     38 : [5 , 50, 0.0, 0.75], hvac off co2 0.75
    #     39 : [5 , 50, 0.0, 1.0] # hvac off co2 1.0
    
    def __init__(self,window_fan_speed=0.5):
        self.is_open = False
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_05 = 21
        self.winter_hvac_on_co2_075 = 22
        self.winter_hvac_on_co2_1 = 23
        
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_05 = 33
        self.summer_hvac_on_co2_75 = 34
        self.summer_hvac_on_co2_1 = 35
        
        self.hvac_off_co2_off = 36
        self.hvac_off_co2_05 = 37 # not used in off controller
        self.hvac_off_co2_075 = 38  # not used in off controller
        self.hvac_off_co2_1 = 39  # not used in off controller
        
        self.window_fan_speed = window_fan_speed
        #Hvac on -off co2 on 
    def select_action(self, state,is_summer,current_step,timesteps_per_hour):
        '''
        Act method for the controller
        '''
        occupancy = state[0][9]
        month = state[0][0]
        #Summer
        if occupancy > 0:

            if is_summer:
                if self.window_fan_speed == 0.5:
                    return self.summer_hvac_on_co2_05
                elif self.window_fan_speed == 0.75:
                    return self.summer_hvac_on_co2_75
                elif self.window_fan_speed == 1.0:
                    return self.summer_hvac_on_co2_1
                elif self.window_fan_speed == 0.0:
                    return self.summer_hvac_on_co2_off
                else:
                    print("Invalid window fan speed")
            else:
                if self.window_fan_speed == 0.5:
                    return self.winter_hvac_on_co2_05
                elif self.window_fan_speed == 0.75:
                    return self.winter_hvac_on_co2_075
                elif self.window_fan_speed == 1.0:
                    return self.winter_hvac_on_co2_1
                elif self.window_fan_speed == 0.0:
                    return self.winter_hvac_on_co2_off
                else:
                    print("Invalid window fan speed")
        else:
            return self.hvac_off_co2_off
        
