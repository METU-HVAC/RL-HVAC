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
from sinergym.utils.wrappers import NormalizeObservation
from environments.environment import create_environment
from utils.dataset import generate_chunks, split_chunks
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
seed = 42  # Set seed for reproducibility
# Set seeds for reproducibility
random.seed(seed)
np.random.seed(seed)

# Ensure the plots directory exists
os.makedirs(plots_dir, exist_ok=True)

# DQN Parameters
state_size =  12 # Adjust based on the size of your observation space
action_size = 71  # 70 discrete actions
gamma = 0.99  # Discount factor
epsilon = 1.0  # Initial exploration rate
epsilon_min = 0.01  # Minimum epsilon
epsilon_decay = (1-0.01)/100  # Epsilon decay rate
batch_size = 64  # Minibatch size for training
train_interval = 100 # Train every n steps
min_buffer_size = 1000  # Start training after buffer fills up
# Initialize the DQN agent
agent = DQNAgent(state_size, action_size,epsilon_decay=epsilon_decay)

variables = [
    'month', 'day_of_month', 'hour', 'outdoor_temperature',
    'outdoor_humidity', 'htg_setpoint', 'clg_setpoint',
    'air_temperature', 'air_humidity', 'people_occupant',
    'HVAC_electricity_demand_rate', 'thermal_comfort_ppd',
    'thermal_comfort_pmv','air_co2','total_electricity_HVAC'
]
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

# Helper to plot and save data
def plot_and_save(outdoor_temps, htg_setpoints, clg_setpoints, air_temps, air_humidities, time_labels, episode_type, episode_num):
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, outdoor_temps, label='Outdoor Temp (°C)', color='blue')
    plt.plot(time_labels, htg_setpoints, label='Heating Setpoint (°C)', color='red')
    plt.plot(time_labels, clg_setpoints, label='Cooling Setpoint (°C)', color='green')
    plt.plot(time_labels, air_temps, label='Air Temp (°C)', color='orange')

    # Add labels, title, and legend
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{episode_type} Episode {episode_num} - Fan Speed {fan_speed}')
    plt.xticks(range(0, len(time_labels), 100), rotation=45)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)

    # Save the plot
    plot_path = f"{plots_dir}/{episode_type}_episode_{episode_num}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved {episode_type} plot for Episode {episode_num} at {plot_path}")

# Helper to run simulation and collect data
def run_simulation(start_date, end_date, episode_type, episode_num):
    env = create_env(start_date, end_date)  # Create a new environment for the chunk

    obs, info = env.reset()
    done = False
    outdoor_temps, htg_setpoints, clg_setpoints, power_consumptions = [], [], [],[]
    air_temps, air_humidities, time_labels = [], [], []
    total_temperature_violation = []
    current_step = 0
    total_reward = 0
    while current_step < steps_per_chunk:
        # hour = int(obs[2])  # Extract 'hour' from observation

        # # Choose action based on the hour
        # if 8 <= hour < 20:
        #     action = [21, 23.5, fan_speed]  # Active mode
        # else:
        #     action = [16, 30, 0.0]  # Default mode
        # action = 0
        # Select action using the DQN policy
        if episode_type == "Training":
            action = agent.choose_action(obs)  # Epsilon-greedy action for training
        else:
            action = agent.choose_greedy_action(obs)  # Greedy action for validation/testing
            print("Action: ",ACTION_MAPPING[action])
        
        next_obs, reward, truncated, terminated, info = env.step(action)
        done = terminated or truncated
        # Only train and store transitions if it's a training episode
        if episode_type == "Training":
            # Store transition in replay buffer
            agent.store_transition(obs, action, reward, next_obs, done)
            # Train DQN every few steps if buffer size is sufficient
            if current_step % train_interval == 0:
                agent.train(current_step)
        obs = next_obs
        # Collect data
        outdoor_temps.append(obs[3])
        htg_setpoints.append(obs[5])
        clg_setpoints.append(obs[6])
        air_temps.append(obs[7])
        air_humidities.append(obs[8])
        power_consumptions.append(obs[11])
        total_temperature_violation.append(info["total_temperature_violation"])
        # Time label
        month, day = int(obs[0]), int(obs[1])
        hour = int(obs[2] + 1)
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
        'air_temps': air_temps,
        'air_humidities': air_humidities,
        'time_labels': time_labels
    }
    env.close()  # Close the environment after use
    return total_reward, sum(power_consumptions),sum(total_temperature_violation),obs_dict
import numpy as np

def calculate_observation_stats(env, num_steps=100000):
    observations = []
    obs, _ = env.reset()
    done = False
    for _ in range(num_steps):
        action = env.action_space.sample()  # Random action to explore environment
        next_obs, reward, truncated, terminated, info= env.step(action)
        observations.append(obs)
        obs = next_obs
        if done:
            obs, _ = env.reset()
    
    # Convert to numpy array and calculate mean, std
    observations = np.array(observations)
    obs_mean = np.mean(observations, axis=0)
    obs_std = np.std(observations, axis=0) + 1e-8  # Add small epsilon to avoid division by zero
    return obs_mean, obs_std
start_date = datetime(1997, 1, 1)
end_date = datetime(1997, 12, 30)
env = create_env(start_date, end_date)  # Create a new environment for the chunk

obs_mean, obs_std = calculate_observation_stats(env)
print("Observation Mean:", obs_mean)
print("Observation Std Dev:", obs_std)
# # Test after training
# print("Test Run")
# test_chunk = random.choice(test_chunks)
# run_simulation(*test_chunk, "Test", 1)

# # Save all episode rewards to a text file
# with open("episode_rewards.txt", "w") as f:
#     for i, reward in enumerate(episode_rewards, 1):
#         f.write(f"Episode {i}: Reward = {reward}\n")
