
import math
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    # mapping = {
    # Summer actions 
    #     20: [20, 23, 0.75, 0.0],
    #     23: [20, 23, 0.75, 0.75],

    # Winter actions
    #     35: [23, 26, 0.75, 0.0],
    #     38: [23, 26, 0.75, 0.75],

    #     45: OFF_ACTION  # Off action
    # }

    def __init__(self):
        self.is_open = False
        self.summer_hvac_on_co2_off = 20
        self.summer_hvac_on_co2_on = 23
        self.winter_hvac_on_co2_off = 35
        self.winter_hvac_on_co2_on = 38

        self.off_action = 45
    def select_action(self, state):
        '''
        Act method for the controller
        '''
        occupancy = state[0][11]
        month_sin = state[0][0]
        month_cos = state[0][1]
        month = int((math.atan2(month_sin, month_cos) * 12 / (2 * math.pi)) % 12) + 1
        #Summer
        if occupancy > 0:
            if month >= 6 and month <= 9:
                return self.summer_hvac_on_co2_on
            else:
                return self.winter_hvac_on_co2_on
        else:
            return self.off_action