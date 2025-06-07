import gymnasium as gym
import sinergym
import matplotlib.pyplot as plt
import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from sinergym.utils.rewards import LinearReward
from typing import Any, Dict, List, Tuple, Union
from sinergym.utils.constants import LOG_REWARD_LEVEL, YEAR
from datetime import datetime
import math
max_energy_penalty = -1.0
#This class combines Co2Reward and the classic reward function
class CO2andTemperatureReward(LinearReward):
    def __init__(
            self,
            co2_variable: str,
            energy_variables: List[str],
            temperature_variables: List[str],
            range_comfort_winter: Tuple[int, int],
            range_comfort_summer: Tuple[int, int],
            ac_energy_weight: float = 0.3,
            fan_energy_weight: float = 0.3,
            co2_weight: float = 0.3,
            temperature_weight: float = 0.3,
            summer_start: Tuple[int, int] = (6, 1),
            summer_final: Tuple[int, int] = (9, 30),
            lambda_energy: float = 1e-2,
            lambda_co2: float = 1.0,
            lambda_temperature: float = 1.0,
            co2_threshold: float = 800,

        ):
            super(LinearReward, self).__init__()
            self.co2_variable = co2_variable
            self.energy_names = energy_variables
            self.temp_names = temperature_variables
            self.W_fan_energy = fan_energy_weight
            self.W_ac_energy = ac_energy_weight
            self.W_co2 = co2_weight
            self.W_temperature = temperature_weight
            self.range_comfort_winter = range_comfort_winter
            self.range_comfort_summer = range_comfort_summer
            self.summer_start = summer_start
            self.summer_final = summer_final

            self.lambda_energy = lambda_energy
            self.lambda_co2 = lambda_co2
            self.lambda_temperature = lambda_temperature
            self.co2_threshold = co2_threshold

            self.energy_rew_arr = []
            self.co2_rew_arr = []
            self.comfort_term_arr = []
            self.daily_timestep_count = 0
            self.timesteps_per_day = 144*9 # 12*24
            

    def _get_seperate_energy_consumed(self, obs_dict):
        """
        Extracts energy consumption values for each variable in energy_names.

        Returns:
            total_energy (float): Sum of all energy values.
            energy_values (dict): Dictionary with individual energy values per variable.
        """
        energy_values = {}
        for name in self.energy_names:
            if name in obs_dict:
                energy_values[name] = obs_dict[name]
            else:
                energy_values[name] = 0.0

        total_energy = sum(energy_values.values())
        return total_energy, energy_values
    def _get_seperate_energy_penalty(self, energy_values):
        """
        Computes energy penalty for each energy variable and returns a combined penalty.
        """
        window_penalty = -energy_values.get("window_fan_energy", 0.0)
        ac_penalty = -energy_values.get("total_electricity_HVAC", 0.0)
        total_penalty = window_penalty + ac_penalty
        return total_penalty, window_penalty, ac_penalty
    def __call__(self, obs_dict: Dict[str, Any]) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate the reward function.

        Args:
            obs_dict (Dict[str, Any]): Dict with observation variable name (key) and observation variable value (value).

        Returns:
            Tuple[float, Dict[str, Any]]: Reward value and dictionary with their individual components.
        """

        try:
            assert all(temp_name in list(obs_dict.keys())
                       for temp_name in self.temp_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the temperature variables specified are not present in observation.')
            raise err
        try:
            assert all(energy_name in list(obs_dict.keys())
                       for energy_name in self.energy_names)
        except AssertionError as err:
            self.logger.error(
                'Some of the energy variables specified are not present in observation.')
            raise err
        try:
            assert self.co2_variable in list(obs_dict.keys())
        except AssertionError as err:
            self.logger.error(
                'CO2 variable specified is not present in observation.')
            raise err
        # Energy penalty
        energy_consumed, energy_values = self._get_seperate_energy_consumed(obs_dict)
        energy_penalty, window_energy_penalty, ac_energy_penalty = self._get_seperate_energy_penalty(energy_values)
       
        # CO2 penalty
        co2_concentration = obs_dict[self.co2_variable]
        co2_reward = self._get_co2_reward(co2_concentration,obs_dict['people_occupant'],self.co2_threshold)

         # Comfort violation calculation
        temp_reward, temp_deviation = self._get_temperature_violation(obs_dict)
        is_occupied = obs_dict['people_occupant'] > 0
        # Weighted sum of both terms
        reward, window_energy_term,ac_energy_term ,co2_term,comfort_term = self._get_reward(window_energy_penalty,ac_energy_penalty, co2_reward, temp_reward,obs_dict['people_occupant'])
        co2_deviation = co2_concentration-self.co2_threshold if (co2_concentration > self.co2_threshold and is_occupied )else 0

        reward_terms = {
            'energy_term': window_energy_term + ac_energy_term,
            'window_energy_term': window_energy_term,
            'ac_energy_term': ac_energy_term,
            'co2_term': co2_term,
            'comfort_term': comfort_term,
            'ac_energy_weight': self.W_ac_energy,
            'fan_energy_weight': self.W_fan_energy,
            'co2_weight': self.W_co2,
            'temperature_weight': self.W_temperature,
            'abs_energy_penalty': energy_penalty,
            'abs_comfort_penalty': temp_deviation if is_occupied else None,
            'abs_co2_penalty': co2_deviation if is_occupied else None,
            'total_power_demand': energy_consumed,
            'total_temperature_violation': temp_reward,
            'co2_concentration': co2_concentration,
            'is_co2_violated': True if is_occupied and co2_deviation > 0 else False if is_occupied else None,
            'is_comfort_violated': True if is_occupied and temp_deviation > 0 else False if is_occupied else None,
            'is_occupied': is_occupied
        }
        return reward, reward_terms
    
    def step_co2_reward(self,co2, limit=800,min_reward = -1):
        if co2 <= limit:
            return 1
        else:
            return min_reward
     
    def co2_penalty_only_reward(self,co2: float, threshold: float = 700.0, max_limit: float = 900.0, min_penalty: float = -5.0) -> float:
        if co2 <= threshold:
            return 1.0
        elif co2 >= max_limit:
            return min_penalty
        else:
            slope = (1.0 - min_penalty) / (max_limit - threshold)
            return 1.0-slope * (co2 - threshold)
    def linear_co2_reward(self,co2_ppm: float, 
                      min_threshold: float = 700, 
                      max_threshold: float = 1300, 
                      max_reward: float = 1.0, 
                      min_reward: float = -5.0) -> float:
        if co2_ppm <= min_threshold:
            return max_reward
        elif co2_ppm >= max_threshold:
            return min_reward
        else:
            slope = (min_reward - max_reward) / (max_threshold - min_threshold)
            return slope * (co2_ppm - min_threshold) + max_reward    

    def _get_co2_reward(self, co2_concentration: float,people_count,co2_limit) -> float:
        if people_count == 0:
            return 0
        return self.co2_penalty_only_reward(co2_concentration)
    # def linear_temp_reward(self,temp, low_limit=23, high_limit=26, tolerance=0.5,min_reward = -1):
    #     if low_limit + tolerance < temp < high_limit - tolerance:
    #         return 1
    #     elif temp < low_limit + tolerance:   
    #         return max((1/tolerance)*(temp - (low_limit+tolerance))+1, min_reward)
    #     elif temp > high_limit - tolerance:
    #         return max(-(1/tolerance)*(temp - (high_limit-tolerance))+1, min_reward)
    #     else:
    #         return 1  # on the limits
    def updated_linear_temp_reward(self,temp, low_limit=23.0, high_limit=26.0, max_penalty=-5.0):
        """
        Reward is 1 between limits.
        Linear penalty as you move away from the limits, with 1 degree deviation = -1 reward.
        Reward saturates at max_penalty.
        """
        if low_limit <= temp <= high_limit:
            return 1.0
        elif temp < low_limit:
            penalty = (temp - low_limit)
        else:
            penalty = (high_limit - temp)
        
        reward = 1 + penalty  # 1 degree deviation = -1 reward
        return max(reward, max_penalty)
    def temperature_comfort_reward(self,T, T_min=23.0, T_max=26.0, people_count=1):
        """
        Calculates a reward based on temperature comfort.

        Rewards -1 for temperatures outside [T_min - 0.5, T_max + 0.5].
        Rewards 0 for temperatures within [T_min, T_max].
        Linearly interpolates between 0 and -1 for temperatures between
        [T_min - 0.5, T_min] and [T_max, T_max + 0.5].
        Returns 0 if there are no people.

        Args:
            T (float): The temperature.
            T_min (float): The minimum comfortable temperature.
            T_max (float): The maximum comfortable temperature.
            people_count (int): The number of people present.

        Returns:
            float: The reward value.
        """
        if people_count == 0:
            return 0  # No occupants, temperature comfort is not relevant
        return self.updated_linear_temp_reward(T, T_min, T_max)


    def _get_temperature_violation(self, obs_dict: Dict[str, Any]) -> Tuple[float, List[float]]:
        """
        Calculate the total temperature violation (ºC) in the current observation.

        Returns:
            Tuple[float, List[float]]: Total temperature violation (ºC) and list with temperature violation in each zone.
        """
        # Extract month and reconstruct day if necessary
        
        month = obs_dict['month']
        day = obs_dict['day_of_month']
        year = YEAR
        current_dt = datetime(int(year), int(month), int(day))

        # Periods
        summer_start_date = datetime(
            int(year),
            self.summer_start[0],
            self.summer_start[1])
        summer_final_date = datetime(
            int(year),
            self.summer_final[0],
            self.summer_final[1])

        if current_dt >= summer_start_date and current_dt <= summer_final_date:
            temp_range = self.range_comfort_summer
        else:
            temp_range = self.range_comfort_winter

        # Process temperature values
        temp_values = [v for k, v in obs_dict.items() if k in self.temp_names]
        total_temp_violation = 0.0
        temp_violations = []
        person_count = obs_dict['people_occupant']
        total_reward = 0
        for T in temp_values:
            reward = self.temperature_comfort_reward(T, T_min=temp_range[0], T_max=temp_range[1], people_count=person_count)
            total_reward += reward

            if person_count > 0 and reward < 0:
                temp_violation = min(abs(temp_range[0] - T), abs(T - temp_range[1]))
                temp_violations.append(temp_violation)
                total_temp_violation += temp_violation

        return total_reward, total_temp_violation
    def _get_reward(self, window_energy_penalty:float,ac_energy_penalty: float, co2_penalty: float,temperature_penalty: float,occupancy: float) -> Tuple[float, float, float,float]:
        """
        Calculate the reward value using penalties for energy, CO2 and temperature.

        Args:
            energy_penalty (float): Negative absolute energy penalty value.
            co2_penalty (float): Negative absolute CO2 penalty value.
            temperature_penalty (float): Negative absolute temperature penalty value.

        Returns:
            Tuple[float, float, float,float]: Total reward, energy term, CO2 term, temperature term.
        """
        #print before the reward
        # global max_energy_penalty
        # if energy_penalty < max_energy_penalty:
        #     max_energy_penalty = energy_penalty
            #print("Max energy penalty: ",max_energy_penalty)
        #print("Energy penalty: ",energy_penalty, "CO2 penalty: ",co2_penalty, "Temperature penalty: ",temperature_penalty)
        #energy_term = self.lambda_energy * self.W_energy * energy_penalty
        
        window_energy_term = self.lambda_energy*100 * self.W_fan_energy * window_energy_penalty
        ac_energy_term = self.lambda_energy * self.W_ac_energy * ac_energy_penalty
        
        if occupancy == 0 and window_energy_term < 0:
            window_energy_term = window_energy_term*2
            
        if occupancy == 0 and ac_energy_term <0:
            ac_energy_term = ac_energy_term*2
        
        co2_term = self.lambda_co2 * self.W_co2 * co2_penalty
        temperature_term = self.lambda_temperature * self.W_temperature * temperature_penalty
        
        
        reward = window_energy_term + ac_energy_term + co2_term + temperature_term
        #print("Total reward: ",reward, "Energy term: ",energy_term, "CO2 term: ",co2_term, "Comfort term: ",temperature_term)
        # self.energy_rew_arr.append(energy_term)
        # self.co2_rew_arr.append(co2_term)
        # self.comfort_term_arr.append(temperature_term)
        #print("E: ",energy_term," C: ",co2_term," T: ",temperature_term)

        # Increment daily timestep count
        self.daily_timestep_count += 1
        
        # # Log and reset at the end of the day
        # if self.daily_timestep_count == self.timesteps_per_day:
        #     avg_energy_reward = sum(self.energy_rew_arr) / len(self.energy_rew_arr)
        #     avg_co2_reward = sum(self.co2_rew_arr) / len(self.co2_rew_arr)
        #     avg_temperature_reward = sum(self.comfort_term_arr) / len(self.comfort_term_arr)

        #     # print(f"WCO₂: {self.W_co2},WE {self.W_energy},WT: {self.W_temperature}")
        #     # print(f"Average Energy Reward for the Day: {avg_energy_reward}")
        #     # print(f"Average CO₂ Reward for the Day: {avg_co2_reward}")
        #     # print(f"Average Temperature Reward for the Day: {avg_temperature_reward}")

        #     self.energy_rew_arr.clear()
        #     self.co2_rew_arr.clear()
        #     self.daily_timestep_count = 0
        
        return reward, window_energy_term,ac_energy_term, co2_term, temperature_term
