
import math
import sys
class WindowOnOffController():
    '''
    OnOffController class opens the window when there is an occupation, else closes.
    '''
    def __init__(self):
        self.is_open = False
    

        
        
        
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_on = 33
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_on = 21

        self.off_action = 36
        #Hvac on -off co2 on 
    def select_action(self, state, is_summer,current_step, timesteps_per_hour):
        '''
        Act method for the controller
        '''
        occupancy = state[0][9]  # Occupancy state, should be > 0 when there's occupancy
        month = int(state[0][0])  # Month (to determine summer or winter)
        day = int(state[0][1])  # Day of the month
        hour = int(state[0][2])  # Hour of the day
        
        # Calculate the current minute based on the current_step
        minute = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))
        #Summer
        if occupancy > 0:

            if is_summer:
                selected_action = self.summer_hvac_on_co2_on
            else:
                selected_action = self.winter_hvac_on_co2_on
        else:
            selected_action = self.off_action
        
        # Log the selected action along with the time details
        time_label = f"{month:02}-{day:02} {hour:02}:{minute:02}"
        print(f"Selected Action: {selected_action} at {time_label}")
        return selected_action
        

class WindowScheduleController():
    """
    When occupied:
      • HVAC always ON.
      • Window (CO₂ fan) OPEN for the first 10minutes of each hour.
    Otherwise, everything OFF.
    """
    def __init__(self):
        self.is_open = False
        self.summer_hvac_on_co2_off = 32
        self.summer_hvac_on_co2_on = 33
        self.winter_hvac_on_co2_off = 20
        self.winter_hvac_on_co2_on = 21

        self.off_action = 36
        self.last_occupation_time = None  # To track when occupation started

    def select_action(self, state,is_summer, current_step, timesteps_per_hour):
        """
        When occupied:
        • HVAC always ON.
        • Window (CO₂ fan) OPEN for the first 10 minutes of each hour.
        Otherwise, everything OFF.
        """
        # --- 1) Decode inputs ----
        occupancy = state[0][9]   # >0 if occupied
        month     = int(state[0][0])
        hour      = int(state[0][2])
        minute    = int((current_step % timesteps_per_hour) * (60 / timesteps_per_hour))

        # Prepare time‐label for debugging
        time_label = f"{month:02d}-{hour:02d}h:{minute:02d}m (step {current_step})"

        # --- 2) Decide action ----
        if occupancy == 0:
            selected_action = self.off_action
            debug_msg = f"[UNoccupied] → OFF"
        else:
            # decide whether window is open
            is_window_open = (minute < 10)
            season = "summer" if is_summer else "winter"

            if is_window_open:
                # first 10 minutes → open window + HVAC
                if season == "summer":
                    selected_action = self.summer_hvac_on_co2_on
                else:
                    selected_action = self.winter_hvac_on_co2_on
                debug_msg = f"[{season.capitalize()} | Window OPEN]"
            else:
                # rest of hour → window closed, but keep HVAC on
                if season == "summer":
                    selected_action = self.summer_hvac_on_co2_off
                else:
                    selected_action = self.winter_hvac_on_co2_off
                debug_msg = f"[{season.capitalize()} | Window CLOSED]"

        # --- 3) Print & flush so you definitely see it ----
        # print(f"{time_label} | Occ={occupancy} → {debug_msg} | Action={selected_action}", flush=True)
        # sys.stdout.flush()

        return selected_action
    
