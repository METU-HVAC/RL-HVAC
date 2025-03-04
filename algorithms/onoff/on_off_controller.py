
import math
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    # mapping = {
    # Winter actions 
    #     28 : [22, 23, 0.75, 0.0],
    #     30 : [22, 23, 0.75, 0.75],
    
        #4 : [20, 21, 0.75, 0.0],
        #6 : [20, 21, 0.75, 0.75],
    # Summer actions
    #     40 : [23, 24, 0.75, 0.0],
    #     42 : [23, 24, 0.75, 0.75],
    
        # 64 : [25, 26, 0.75, 0.0],
        # 66 : [25, 26, 0.75, 0.75],

    #     72: [5,50,0.0,0.0],  # Off action
    #     74: [5,50,0.0,0.75], Hvac off co2 on
    # }

        # 0 : [20,22,1.0,0.0],
        # 1 : [20,22,1.0,1.0],
        # 2 : [22,24,1.0,0.0],
        # 3 : [22,24,1.0,1.0],
        # 4 : [24,26,1.0,0.0],
        # 5 : [24,26,1.0,1.0],
        # 6 : [5,50,0.0,0.0],
        # 7 : [5,50,0.0,1.0],

    def __init__(self):
        self.is_open = False
        self.summer_hvac_on_co2_off = 4
        self.summer_hvac_on_co2_on = 5
        self.winter_hvac_on_co2_off = 0
        self.winter_hvac_on_co2_on = 1

        self.off_action = 6
        #Hvac on -off co2 on 
    def select_action(self, state):
        '''
        Act method for the controller
        '''
        occupancy = state[0][9]
        month = state[0][0]
        #Summer
        if occupancy > 0:

            if month >= 6 and month <= 9:
                return self.summer_hvac_on_co2_on
            else:
                return self.winter_hvac_on_co2_on
        else:
            return self.off_action