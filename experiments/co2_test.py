import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
from algorithms.dqn.dqn import *
from environments.reward import *
import torch
from sinergym.utils.wrappers import NormalizeObservation
from common.utils import *
from environments.environment import create_environment
from utils.dataset import generate_chunks, split_chunks
from utils.visualization import plot_and_save
from tqdm import tqdm

# Configuration
timesteps_per_hour = 12  # 5-minute intervals
days_per_chunk = 10
timestep_per_day = timesteps_per_hour * 24
steps_per_chunk = timestep_per_day * days_per_chunk
fan_speed = 0.9  # Float between 0 and 1
num_episodes = 100  # Total episodes for training
validation_interval = 3  # Validate every 2 episodes
plots_dir = "results/plots/dqn"  # Directory to store plots
extra_params = {
    'timesteps_per_hour': timesteps_per_hour,
    'runperiod':(1,1,1997,12,3,1997)  # Full year simulation
}
obs_mean = [6.2457600e+00 ,1.5368000e+01, 1.1494040e+01 ,2.2182905e+01 ,3.5119591e+01,
 2.0017839e+01 ,2.5732735e+01, 2.2225080e+01, 3.1512318e+01, 1.6787984e+00,
 4.7591147e+02, 1.5643309e+05]
obs_std_dev =  [3.2992606e+00, 8.7997694e+00, 6.9240804e+00, 8.8425007e+00 ,2.4141937e+01,
 3.7141514e+00 ,3.7083476e+00, 2.9696338e+00 ,1.5955880e+01, 1.9859594e+00,
 6.8838965e+02 ,2.0938520e+05]

obs_mean = torch.tensor(obs_mean, dtype=torch.float32,device=device)
obs_std_dev = torch.tensor(obs_std_dev, dtype=torch.float32,device=device)
seed = 42  # Set seed for reproducibility
# Set seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

remove_previous_run_logs()
        
state_size =  12 # Adjust based on the size of your observation space
action_size = 57  # 70 discrete actions
train_interval = 100 # Train every n steps


# Observation variables for clarity
variables = [
    'month', 'day_of_month', 'hour', 'outdoor_temperature',
    'outdoor_humidity', 'htg_setpoint', 'clg_setpoint',
    'air_temperature', 'air_humidity', 'people_occupant',
    'HVAC_electricity_demand_rate', 'total_electricity_HVAC'
]
# Helper to normalize an observation
def normalize_observation(obs, mean, std):
    return (obs - mean) / std

# Define parameters
start_date = datetime(1997, 1, 1)
days_per_chunk = 10
total_days = 365
# Generate and split chunks
chunks = generate_chunks(start_date, days_per_chunk, total_days)
train_chunks, val_chunks, test_chunks = split_chunks(chunks, train_ratio=0.7, val_ratio=0.2, seed=seed)

num_episodes = 10  # Total number of episodes (full sweeps through the dataset)
total_number_of_training_chunks = len(train_chunks)
total_training_steps = total_number_of_training_chunks * num_episodes*steps_per_chunk
current_training_step = 0

# Initialize the DQN agent
agent = DQNAgent(state_size, action_size,total_training_steps)
# Helper to run simulation and collect data

# Create the main experiment directory (only once)
experiment_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
experiment_dir = os.path.join(plots_dir, f"experiment_{experiment_timestamp}")
os.makedirs(experiment_dir, exist_ok=True)

def run_simulation(start_date, end_date, episode_type, episode_num):
    env = create_environment(start_date, end_date,MyCustomReward)  # Create a new environment for the chunk
    state, info = env.reset()
    
    data = state
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    
    done = False
    outdoor_temps, htg_setpoints, clg_setpoints, power_consumptions = [], [], [],[]
    air_temps, air_humidities, time_labels,fan_speeds = [], [], [],[]
    total_temperature_violation = []
    current_step = 0
    total_reward = 0
    while current_step < steps_per_chunk:
        if state is None:
            print("State is None")
        state = normalize_observation(state,obs_mean,obs_std_dev)
        if episode_type == "Training":
            action = agent.select_action(state)  # Epsilon-greedy action for training
        else:
            action = agent.choose_greedy_action(state)  # Greedy action for validation/testing
        fan_speed = ACTION_MAPPING[action.item()][2]
        np_action = np.array([action.item()], dtype=np.float32)  # Adjust dtype to match environment
        observation, reward, truncated, terminated, info = env.step(np_action)

        #observation, reward, truncated, terminated, info = env.step(action.item())
        data = observation
        done = terminated or truncated
        reward = torch.tensor([reward], device=device)

        if done:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
            next_state = normalize_observation(next_state,obs_mean,obs_std_dev)
        # Only train and store transitions if it's a training episode
        if episode_type == "Training":
            # Store transition in replay buffer
            agent.store_transition(state, action, next_state, reward)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                # Perform one step of the optimization (on the policy network)
                agent.optimize_model()
        state = next_state
        
        # Collect data
        outdoor_temps.append(data[3])
        htg_setpoints.append(data[5])
        clg_setpoints.append(data[6])
        air_temps.append(data[7])
        air_humidities.append(data[8])
        power_consumptions.append(data[11])
        fan_speeds.append(fan_speed)
        total_temperature_violation.append(info["total_temperature_violation"])
        # Time label
        month, day = int(data[0]), int(data[1])
        hour = int(data[2] + 1)
        minute = (current_step % timesteps_per_hour) * (60 / timesteps_per_hour)
        time_labels.append(f"{month:02}-{day:02} {hour:02}:{minute:02}")
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
        'time_labels': time_labels
    }
    env.close()  # Close the environment after use
    return total_reward, sum(power_consumptions),sum(total_temperature_violation),obs_dict



model_dir = "results/models/dqn"  # Directory to store the best model

# Training, Validation, and Test Loop
for episode in range(1, num_episodes + 1):
    # Initialize progress bar for the episode
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
        for train_chunk in train_chunks:
            reward , power_consumption,temp_viol,obs_dict = run_simulation(*train_chunk,
                                                                    "Training", 
                                                                    episode)
            train_total_reward += reward
            train_total_power += power_consumption
            pbar.set_postfix_str(f"Training Chunk {pbar.n + 1}/{len(train_chunks)}")
            pbar.update(1)
            current_training_step += 1
            
            #Decay epsilon every total/100 steps
            
        avg_train_reward = (train_total_reward / len(train_chunks)).item()

        with open("train_average_reward.txt", "a") as f:
            f.write(f"Episode {episode}: Train Average Reward = {avg_train_reward} ,Epsilon = {agent.eps_threshold}\n")
        plot_and_save(**obs_dict, episode_type="Training", episode_num=episode,plots_dir=experiment_dir)

        # Validation: Full sweep over the validation dataset (without shuffling)
        val_total_reward = 0
        val_total_power = 0
        val_temp_violation = 0
        for val_chunk in val_chunks:
            reward, power_consumption ,temp_viol,obs_dict= run_simulation(*val_chunk, "Validation", episode)
            val_total_reward += reward
            val_total_power += power_consumption
            val_temp_violation +=temp_viol
            pbar.set_postfix_str(f"Validation Chunk {pbar.n + 1}/{len(train_chunks) + len(val_chunks)}")
            pbar.update(1)
        # Calculate average reward and power consumption for validation
        avg_val_reward = (val_total_reward / len(val_chunks)).item()
        
        # Save the best model if necessary
        save_best_model(agent, episode, avg_val_reward, model_dir)
        avg_val_power = val_total_power / len(val_chunks)
        avg_val_temp_violation = val_temp_violation / len(val_chunks)
        with open("val_average_reward.txt", "a") as f:
            f.write(f"Episode {episode}: Val Avg Reward = {avg_val_reward}, Val Avg Power Conspt = {avg_val_power}, Val Avg Temp Violation = {avg_val_temp_violation}\n")
        plot_and_save(**obs_dict, episode_type="Validation", episode_num=episode,plots_dir=experiment_dir)
        # Update progress bar to reflect final validation averages
        pbar.set_postfix(
            {
                "TrR": f"{avg_train_reward:.1f}",
                "ValR":f"{avg_val_reward:.1f}",
                "AvgPwr":f"{avg_val_power:.1f}", 
                "AvgTempViol": f"{avg_val_temp_violation:.1f}",
            }
        )
