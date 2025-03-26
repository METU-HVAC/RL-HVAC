import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
from algorithms.dqn.dqn import *
from algorithms.rbc.rbc import *
from environments.reward import *
from environments.environment import CO2_AND_TEMP_REWARD_CONFIG
import torch
from sinergym.utils.wrappers import DatetimeWrapper

from environments.environment import create_environment
from utils.dataset import generate_chunks, split_chunks
from utils.visualization import plot_and_save, plot_csv_data
from tqdm import tqdm
import wandb
import pandas as pd
from utils.experiment_utils import create_experiment_name, save_run_metrics, calculate_time_label

ENV_NAME = "A403"
ALGORITHM_NAME = "PPO_TRIAL"
NUM_EPISODES = 10

def run_simulation(start_date, end_date, episode_type, steps_per_chunk,agent,train_interval,timesteps_per_hour,reward_config):
    env = create_environment(start_date, end_date,CO2andTemperatureReward,timesteps_per_hour=timesteps_per_hour,reward_kwargs=reward_config)  # Create a new environment for the chunk
    state, info = env.reset()
    
    data = state
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False
    outdoor_temps, htg_setpoints, clg_setpoints, power_consumptions = [], [], [],[]
    air_temps, air_humidities, time_labels,fan_speeds = [], [], [],[]
    total_temperature_violation,people_occupants,co2_levels = [], [], []
    temp_violations,co2_violations = [],[]
    current_step = 0
    total_reward = 0
    loss_list = []
    while current_step < steps_per_chunk:
        if state is None:
            print("State is None")
        #state = normalize_observation(state,obs_mean,obs_std_dev)
       
       
        if episode_type == "Training":
            action = agent.select_action(state)  # Epsilon-greedy action for training
        else:
            action = agent.choose_greedy_action(state)  # Greedy action for validation/testing

        fan_speed = DEFAULT_A403_DISCRETE_FUNCTION(action.item())[3]
        np_action = np.array([action.item()], dtype=np.float32)  # Adjust dtype to match environment
        observation, reward, truncated, terminated, info = env.step(np_action)

        #observation, reward, truncated, terminated, info = env.step(action.item())
        data = observation
        done = terminated or truncated
        reward = torch.tensor([reward],dtype=torch.float32, device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            #next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        if episode_type == "Training":
            # Store transition in replay buffer
            agent.store_transition(state, action, next_state, reward)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                # Perform one step of the optimization (on the policy network)
                loss = agent.optimize_model()
                if loss is not None:
                    loss_list.append(loss)
        state = next_state
        variables = [
            'month_sin', 'month_cos', 'is_weekend','hour_sin','hour_cos', 'outdoor_temperature',
            'outdoor_humidity', 'htg_setpoint', 'clg_setpoint',
            'air_temperature', 'air_humidity', 'people_occupant',
            'HVAC_electricity_demand_rate', 'thermal_comfort_ppd',
            'thermal_comfort_pmv','air_co2','total_electricity_HVAC'
        ]
        outdoor_temps.append(data[5])
        htg_setpoints.append(data[7])
        clg_setpoints.append(data[8])
        air_temps.append(data[9])
        air_humidities.append(data[10])
        people_occupants.append(data[11])
        power_consumptions.append(data[12])
        co2_levels.append(data[15])
        fan_speeds.append(fan_speed)
        
        total_temperature_violation.append(data[15])

        temp_violations.append(info['is_comfort_violated'])
        co2_violations.append(info['is_co2_violated'])
        # # Time label
        # month, day = int(data[0]), int(data[1])
        # hour = int(data[2] + 1)
        # minute = (current_step % timesteps_per_hour) * (60 / timesteps_per_hour)
        time_label = calculate_time_label(data[0], data[1], data[3], data[4], current_step, timesteps_per_hour)
        #time_labels.append(f"{month:02}-{day:02} {hour:02}:{minute:02}")
        time_labels.append(time_label)
        total_reward += reward
        #print(info)
        
        if done:
            env.reset()

        current_step += 1
    
    # Save the lists into a dictinary and return them, plot later.
    obs_dict = {
        'outdoor_temps': outdoor_temps,
        'htg_setpoints': htg_setpoints,
        'clg_setpoints': clg_setpoints,
        'fan_speeds': fan_speeds,
        'air_temps': air_temps,
        'air_humidities': air_humidities,
        'time_labels': time_labels,
        'total_temperature_violation': total_temperature_violation,
        'power_consumptions': power_consumptions,
        'people_occupants': people_occupants,
        'co2_levels': co2_levels,
        'temp_violations': temp_violations,
        'co2_violations': co2_violations
    }
    env.close()  # Close the environment after use
    if len(loss_list) > 0:
        loss = sum(loss_list) / len(loss_list)
    else:
        loss = 0
    return total_reward, sum(power_consumptions),sum(total_temperature_violation),loss,obs_dict

def train(config=None):
    with wandb.init(config=config):
        config = wandb.config
        
        seed = 42  # Set seed for reproducibility
        # Set seeds for reproducibility
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        remove_previous_run_logs()
                
        state_size =  17 # Adjust based on the size of your observation space
        action_size = 37  
        train_interval = 100 # Train every n steps
        timesteps_per_hour = 6  # 10-minute intervals
        days_per_chunk = 10
        timestep_per_day = timesteps_per_hour * 24
        steps_per_chunk = timestep_per_day * days_per_chunk
        num_episodes = 10  # Total episodes for training
        start_date = datetime(1997, 1, 1)
        days_per_chunk = 10
        total_days = 365
        plots_dir = "results/plots/dqn"  # Directory to store plots
        # Observation variables for clarity
        variables = [
            'month_sin', 'month_cos', 'is_weekend','hour_sin','hour_cos', 'outdoor_temperature',
            'outdoor_humidity', 'htg_setpoint', 'clg_setpoint',
            'air_temperature', 'air_humidity', 'people_occupant',
            'HVAC_electricity_demand_rate', 'thermal_comfort_ppd',
            'thermal_comfort_pmv','air_co2','total_electricity_HVAC'
        ]
        extra_params = {
            'timesteps_per_hour': timesteps_per_hour,
            'runperiod':(1,1,1997,12,3,1997)  # Full year simulation
        }

        # Generate and split chunks
        chunks = generate_chunks(start_date, days_per_chunk, total_days)
        train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.8, val_ratio=0.2, seed=seed)

        num_episodes = NUM_EPISODES  # Total number of episodes (full sweeps through the dataset)  
        total_number_of_training_chunks = len(train_chunks)
        total_number_of_test_chunks = len(test_chunks)

        total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
        total_testing_steps = total_number_of_test_chunks * num_episodes*steps_per_chunk
        current_training_step = 0
        training_config = {
            "batch_size": 64,
            "gamma": 0.99,
            "eps_start": 0.9,
            "eps_end": 0.05,
            "eps_decay": 5,
            "tau": 0.005,
            "lr":1e-4,
            "memory_capacity": 100000
        }
        reward_config = {
            'temperature_variables': ['air_temperature'],
            'co2_variable': 'air_co2',
            'energy_variables': ['HVAC_electricity_demand_rate'],
            'range_comfort_winter': (20.0, 23.5),
            'range_comfort_summer': (23.0, 26.0),
            'energy_weight': config.energy_weight,
            'co2_weight': config.co2_weight,
            'temperature_weight': config.temp_weight,
            'lambda_energy': 1e-2,
            'lambda_temperature': 1.0,
            'lambda_co2': 1.0,
            'co2_threshold': 700,
        }

        # Initialize the DQN agent
        agent = DQNAgent(state_size, action_size,total_training_steps,training_config)
        # Create the main experiment directory (only once)
        experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        experiment_dir = os.path.join(plots_dir, f"experiment_{experiment_timestamp}")
        os.makedirs(experiment_dir, exist_ok=True)
        
        for episode in range(1, num_episodes + 1):
            print("Epsilon of the model:" ,agent.eps_threshold)
            with tqdm(total=len(train_chunks) + len(val_chunks), 
                    desc=f"Episode {episode}", 
                    ncols=120, 
                    unit="chunk", 
                    leave=True) as pbar:
                # Shuffle train chunks at the start of every episode
                random.shuffle(train_chunks)
                # Training: Full sweep over the shuffled training dataset
                train_total_reward = 0
                train_total_power = 0
                train_total_temp_violation = 0
                train_total_co2_concentration = []
                train_total_occupancy_co2_concentration = []
                co_violation_percentage = 0
                temp_violation_percentage = 0
                total_loss_list = []
                co2_levels , inside_temp_levels, outside_temp_levels = [],[],[]
                for train_chunk in train_chunks:
                    reward , power_consumption,temp_viol,loss,obs_dict = run_simulation(*train_chunk,
                                                                            "Training", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                    
                    total_loss_list.append(loss)
                    train_total_reward += reward
                    train_total_power += power_consumption
                    train_total_temp_violation += temp_viol
                    pbar.set_postfix_str(f"Train Chunk {pbar.n + 1}/{len(train_chunks)}")
                    pbar.update(1)
                    current_training_step += 1
                    
                    for i in range(len(obs_dict['co2_levels'])):
                        co2_concentration = obs_dict['co2_levels'][i]
                        train_total_co2_concentration.append(co2_concentration)
                        if obs_dict['people_occupants'][i] != 0:
                            train_total_occupancy_co2_concentration.append(co2_concentration)
                    for i in range(len(obs_dict['co2_levels'])):
                        co2_levels.append(obs_dict['co2_levels'][i])
                        inside_temp_levels.append(obs_dict['air_temps'][i])
                        outside_temp_levels.append(obs_dict['outdoor_temps'][i])
                    co_violation_percentage += sum(obs_dict['co2_violations'])/len(obs_dict['co2_violations'])*100
                    temp_violation_percentage += sum(obs_dict['temp_violations'])/len(obs_dict['temp_violations'])*100 
                    
                    
                    
                print(f"Loss for episode {episode}: {np.mean(total_loss_list)}")    
                avg_train_reward = (train_total_reward / len(train_chunks)).item()
                avg_train_power = (train_total_power / len(train_chunks))
                
                avg_train_temp_violation = (temp_violation_percentage / len(train_chunks))
                avg_train_co2_violation = (co_violation_percentage / len(train_chunks))
                
                
                with open("train_average_reward.txt", "a") as f:
                    f.write(f"Episode {episode}: Train Average Reward = {avg_train_reward} ,Epsilon = {agent.eps_threshold}\n")
                #plot_and_save(**obs_dict, episode_type="Training", episode_num=episode,plots_dir=experiment_dir)
            
                # Validation: Full sweep over the shuffled validation dataset
                val_total_reward = 0
                val_total_power = 0
                val_total_temp_violation = 0
                val_total_co2_concentration = []
                co_violation_percentage = 0
                temp_violation_percentage = 0
                val_total_occupancy_co2_concentration = []
                test_count = 0
                for val_chunk in val_chunks:
                    test_count += 1
                    reward , power_consumption,temp_viol,loss,obs_dict = run_simulation(*val_chunk,
                                                                            "Validation", 
                                                                            steps_per_chunk,
                                                                            agent,
                                                                            train_interval,
                                                                            timesteps_per_hour,
                                                                            reward_config)
                    
                    
                    
                    val_total_reward += reward
                    val_total_power += power_consumption
                    val_total_temp_violation += temp_viol
                    pbar.set_postfix_str(f"val Chunk {pbar.n + 1}/{len(val_chunks)}")
                    pbar.update(1)
                    
                    
                    for i in range(len(obs_dict['co2_levels'])):
                        co2_concentration = obs_dict['co2_levels'][i]
                        val_total_co2_concentration.append(co2_concentration)
                        if obs_dict['people_occupants'][i] != 0:
                            val_total_occupancy_co2_concentration.append(co2_concentration)

                avg_val_reward = (val_total_reward / len(val_chunks)).item()
                avg_val_power = (val_total_power / len(val_chunks))
                avg_val_temp_violation = (temp_violation_percentage / len(val_chunks))
                avg_val_co2_violation = (co_violation_percentage / len(val_chunks))

                wandb.log({"avg_power": avg_train_power},step=episode * len(inside_temp_levels))
                wandb.log({"avg_co2_train":np.mean(train_total_co2_concentration),"avg_occupancy_co2_train":np.mean(train_total_occupancy_co2_concentration)},step=episode * len(inside_temp_levels))
                wandb.log({"avg_reward_train":avg_train_reward},step=episode * len(inside_temp_levels))
                wandb.log({"avg_train_temp_violation_percentage":avg_train_temp_violation},step=episode * len(inside_temp_levels))
                wandb.log({"avg_train_co2_violation_percentage":avg_train_co2_violation},step=episode * len(inside_temp_levels))
                wandb.log({"avg_val_power": avg_val_power},step=episode * len(inside_temp_levels))
                wandb.log({"avg_co2_val":np.mean(val_total_co2_concentration),"avg_occupancy_co2_val":np.mean(val_total_occupancy_co2_concentration)},step=episode * len(inside_temp_levels))
                wandb.log({"avg_reward_val":avg_val_reward},step=episode * len(inside_temp_levels))
                wandb.log({"avg_val_temp_violation_percentage":avg_val_temp_violation},step=episode * len(inside_temp_levels))
                wandb.log({"avg_val_co2_violation_percentage":avg_val_co2_violation},step=episode * len(inside_temp_levels))

                for timestep in range(len(inside_temp_levels)):
                    wandb.log({
                        "val_inside_temperature_timestep": inside_temp_levels[timestep],
                        "val_outside_temperature_timestep": outside_temp_levels[timestep],
                        "val_co2_level_timestep": co2_levels[timestep],
                    }, step=episode * len(inside_temp_levels) + timestep)  # Increment step for each timestep
                    wandb.log({
                        "train_inside_temperature_timestep": inside_temp_levels[timestep],
                        "train_outside_temperature_timestep": outside_temp_levels[timestep],
                        "train_co2_level_timestep": co2_levels[timestep],
                    }, step=episode * len(inside_temp_levels) + timestep)  # Increment step for each timestep
                
                with open("val_average_reward.txt", "a") as f:
                    f.write(f"Episode {episode}: val Average Reward = {avg_val_reward} ,Epsilon = {agent.eps_threshold}\n")
                #plot_and_save(**obs_dict, episode_type="Validation", episode_num=episode,plots_dir=experiment_dir)
                
                
                # Update progress bar to reflect final validation averages
                pbar.set_postfix(
                    {
                        "TrR": f"{avg_train_reward:.1f}",
                        "ValR":f"{avg_val_reward:.1f}",
                        "AvgPwr":f"{avg_val_power:.1f}", 
                        "AvgCo2OccConc": f"{np.mean(val_total_occupancy_co2_concentration):.1f}",
                    }
                )
        ##AFTER TRAINING
        save_run_metrics(experiment_dir,avg_val_reward, avg_val_power, 
                np.mean(val_total_co2_concentration),np.mean(val_total_occupancy_co2_concentration))
        # Save data to CSV for visualization
        df = pd.DataFrame({
            'Time': obs_dict['time_labels'],
            'Outdoor_Temperature': obs_dict['outdoor_temps'],
            'Heating_Setpoint': obs_dict['htg_setpoints'],
            'Cooling_Setpoint': obs_dict['clg_setpoints'],
            'Air_Temperature': obs_dict['air_temps'],
            'Air_Humidity': obs_dict['air_humidities'],
            'Power_Consumption': obs_dict['power_consumptions'],
            'Fan_Speed': obs_dict['fan_speeds'],
            'Temperature_Violation': obs_dict['total_temperature_violation'],
            'CO2_Level': obs_dict['co2_levels'],
            'People_Occupants': obs_dict['people_occupants']
        })


        # Save DataFrame to CSV
        file_path = os.path.join(experiment_dir, "dqn_data.csv")
        df.to_csv(file_path, index=False)

        



# name= create_experiment_name(env_name=ENV_NAME, episodes=NUM_EPISODES,algorithm_name=ALGORITHM_NAME)
# sweep_config = {
#     'method': 'bayes' ,
#     'name' : name
#     }
# metric = {
#     'name': 'avg_power',
#     'goal': 'minimize'   
#     }
# parameters_dict = ({
#     'energy_weight': {
#         'distribution': 'uniform',
#         'min': 0.1,
#         'max': 1.0
#       },
#     'co2_weight': {
#         'distribution': 'uniform',
#         'min': 0.1,
#         'max': 1.0
#       },
#     'temp_weight': {
#         'distribution': 'uniform',
#         'min': 0.1,
#         'max': 1.0
#       }
#     })
# sweep_config['parameters'] = parameters_dict
# sweep_config['metric'] = metric


# sweep_id = wandb.sweep(sweep_config, project="A403-Train",entity="mehmetbh")
# wandb.agent(sweep_id, train, count=20)

# wandb.finish()

