
import math
class OnOffController():
    '''
    OnOffController class opens the fan when there is an occupation, else closes.
    '''
    # mapping = {
    # Summer actions 
    #     16: [20, 23, 0.75, 0.0],
    #     18: [20, 23, 0.75, 0.75],

    # Winter actions
    #     28: [23, 26, 0.75, 0.0],
    #     30: [23, 26, 0.75, 0.75],

    #     36: OFF_ACTION  # Off action
    # }
    # Extract month and reconstruct day if necessary
    #     month_sin = obs_dict['month_sin']
    #     month_cos = obs_dict['month_cos']
    #     year = YEAR

    #     Reconstruct the month (1-12)
    #     month = int((math.atan2(month_sin, month_cos) * 12 / (2 * math.pi)) % 12) + 1
    #     print("Month: ",month)
    #     If day_of_month is no longer present, you may need an alternative source for it.
    #     day = obs_dict.get('day_of_month', 15)  # Default to mid-month if day isn't available

    #     current_dt = datetime(year, month, day)

    #     Periods
    #     summer_start_date = datetime(year, self.summer_start[0], self.summer_start[1])
    #     summer_final_date = datetime(year, self.summer_final[0], self.summer_final[1])

    #     Determine temperature comfort range based on the season
    #     if summer_start_date <= current_dt <= summer_final_date:
    #         temp_range = self.range_comfort_summer
    #     else:
    #         temp_range = self.range_comfort_winter
    

    def __init__(self):
        self.is_open = False
        self.summer_hvac_on_co2_off = 16
        self.summer_hvac_on_co2_on = 18
        self.winter_hvac_on_co2_off = 28
        self.winter_hvac_on_co2_on = 30

        self.off_action = 36
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