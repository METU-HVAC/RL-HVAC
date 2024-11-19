import sinergym
import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import random
import os
from sinergym.utils.constants import *
from DQN import *
from reward import *
from sinergym.utils.wrappers import NormalizeObservation
import shutil
# Configuration
timesteps_per_hour = 12  # 5-minute intervals
days_per_chunk = 10
timestep_per_day = timesteps_per_hour * 24
steps_per_chunk = timestep_per_day * days_per_chunk
fan_speed = 0.9  # Float between 0 and 1
num_episodes = 100  # Total episodes for training
validation_interval = 3  # Validate every 2 episodes
plots_dir = "plots"  # Directory to store plots
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
#Before starting, clear any directories that may have been created in previous runs
#They start with Eplus-env-* and are created by EnergyPlus


for root, dirs, files in os.walk(".", topdown=False):
    for name in dirs:
        if name.startswith("Eplus-env-"):
            dir_path = os.path.join(root, name)
            shutil.rmtree(dir_path)
        
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
agent = DQNAgent(state_size, action_size)

# Observation variables for clarity
variables = [
    'month', 'day_of_month', 'hour', 'outdoor_temperature',
    'outdoor_humidity', 'htg_setpoint', 'clg_setpoint',
    'air_temperature', 'air_humidity', 'people_occupant',
    'HVAC_electricity_demand_rate', 'total_electricity_HVAC'
]
# Helper to create a new environment with given start and end dates
def create_env(start_date, end_date):
    extra_params = {
        'timesteps_per_hour': timesteps_per_hour,
        'runperiod': (
            start_date.day, start_date.month, start_date.year,
            end_date.day, end_date.month, end_date.year
        )
    }
    env = gym.make('Eplus-A403-hot-discrete-v1', 
                   reward=MyCustomReward,
                   reward_kwargs={
                                'temperature_variables': ['air_temperature'],
                                'energy_variables': ['HVAC_electricity_demand_rate'],
                                'range_comfort_winter': (20.0, 23.5),
                                'range_comfort_summer': (23.0, 26.0),
                                'energy_weight':  0.3,
                                'lambda_energy':  1e-4,
                                'lambda_temperature':1.0,
                                },
                   config_params=extra_params)
    return env
# Helper to normalize an observation
def normalize_observation(obs, mean, std):
    return (obs - mean) / std
# Generate 5-day chunks for the entire year
def generate_chunks():
    chunks = []
    start_date = datetime(1997, 1, 1)
    for i in range(365 // days_per_chunk):
        chunk_start = start_date + timedelta(days=i * days_per_chunk)
        chunk_end = chunk_start + timedelta(days=days_per_chunk - 1)
        chunks.append((chunk_start, chunk_end))
    return chunks

chunks = generate_chunks()
random.shuffle(chunks)  # Shuffling with the set seed ensures reproducibility

# 2. Split chunks into train, validation, and test sets
train_size = int(0.7 * len(chunks))
val_size = int(0.1 * len(chunks))
test_size = len(chunks) - train_size - val_size

train_chunks = chunks[:train_size]
val_chunks = chunks[train_size:train_size + val_size]
test_chunks = chunks[train_size + val_size:]

# 3. Helper to set environment dates
def set_env_runperiod(env, start_date, end_date):
    print(env.get_wrapper_attr('config_params'))
    env.config.runperiod = (
        start_date.day, start_date.month, start_date.year,
        end_date.day, end_date.month, end_date.year
    )
    env.reset()
    

# Helper to plot and save data
def plot_and_save(outdoor_temps, htg_setpoints, clg_setpoints,fan_speeds, air_temps, air_humidities, time_labels, episode_type, episode_num):
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, outdoor_temps, label='Outdoor Temp (°C)', color='blue')
    plt.plot(time_labels, htg_setpoints, label='Heating Setpoint (°C)', color='red')
    plt.plot(time_labels, clg_setpoints, label='Cooling Setpoint (°C)', color='green')
    plt.plot(time_labels, air_temps, label='Air Temp (°C)', color='orange')

    # Add labels, title, and legend
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Temperature (°C)')
    plt.title(f'{episode_type} Episode {episode_num}')
    plt.xticks(range(0, len(time_labels), 100), rotation=45)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)

    # Save the plot
    plot_path = f"{plots_dir}/{episode_type}_episode_{episode_num}.png"
    plt.savefig(plot_path)
    plt.close()

    print(f"Saved {episode_type} plot for Episode {episode_num} at {plot_path}")
    # Plot fan speed data separately
    plt.figure(figsize=(12, 6))
    plt.plot(time_labels, fan_speeds, label='Fan Speed (%)', color='purple')

    # Add labels, title, and legend for fan speed plot
    plt.xlabel('Time (HH:MM)')
    plt.ylabel('Fan Speed (%)')
    plt.title(f'{episode_type} Episode {episode_num} - Fan Speed')
    plt.xticks(range(0, len(time_labels), 100), rotation=45)
    plt.tight_layout()
    plt.legend(loc='best')
    plt.grid(True)

    # Save the fan speed plot
    fan_speed_plot_path = f"{plots_dir}/{episode_type}_episode_{episode_num}_fan_speed.png"
    plt.savefig(fan_speed_plot_path)
    plt.close()
    print(f"Saved {episode_type} fan speed plot for Episode {episode_num} at {fan_speed_plot_path}")

# Helper to run simulation and collect data
def run_simulation(start_date, end_date, episode_type, episode_num):
    env = create_env(start_date, end_date)  # Create a new environment for the chunk

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

        state = normalize_observation(state,obs_mean,obs_std_dev)
        if episode_type == "Training":
            action = agent.select_action(state)  # Epsilon-greedy action for training
        else:
            action = agent.choose_greedy_action(state)  # Greedy action for validation/testing
            print("Action: ",ACTION_MAPPING[action.item()])
        fan_speed = ACTION_MAPPING[action.item()][2]
        observation, reward, truncated, terminated, info = env.step(action.item())
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


num_episodes = 30  # Total number of episodes (full sweeps through the dataset)
total_number_of_training_chunks = len(train_chunks)
total_training_steps = total_number_of_training_chunks * num_episodes
current_training_step = 0
# Training, Validation, and Test Loop
for episode in range(1, num_episodes + 1):
    print(f"Training Episode {episode}")
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
        current_training_step += 1
        
        #Decay epsilon every total/100 steps
        
    avg_train_reward = train_total_reward / len(train_chunks)
    
    with open("train_average_reward.txt", "a") as f:
        f.write(f"Episode {episode}: Train Average Reward = {avg_train_reward} ,Epsilon = {agent.eps_threshold}\n")
    plot_and_save(**obs_dict, episode_type="Training", episode_num=episode)
    
    # Validation: Full sweep over the validation dataset (without shuffling)
    val_total_reward = 0
    val_total_power = 0
    val_temp_violation = 0
    for val_chunk in val_chunks:
        reward, power_consumption ,temp_viol,obs_dict= run_simulation(*val_chunk, "Validation", episode)
        val_total_reward += reward
        val_total_power += power_consumption
        val_temp_violation +=temp_viol
    # Calculate average reward and power consumption for validation
    avg_val_reward = val_total_reward / len(val_chunks)
    avg_val_power = val_total_power / len(val_chunks)
    avg_val_temp_violation = val_temp_violation / len(val_chunks)
    with open("val_average_reward.txt", "a") as f:
        f.write(f"Episode {episode}: Validation Average Reward = {avg_val_reward}\n")
        f.write(f"Episode {episode}: Validation Average Power Consumption = {avg_val_power}\n")
        f.write(f"Episode {episode}: Validation Average Temperature Violation = {avg_val_temp_violation}\n")
    plot_and_save(**obs_dict, episode_type="Validation", episode_num=episode)

# # Test after training
# print("Test Run")
# test_chunk = random.choice(test_chunks)
# run_simulation(*test_chunk, "Test", 1)

# # Save all episode rewards to a text file
# with open("episode_rewards.txt", "w") as f:
#     for i, reward in enumerate(episode_rewards, 1):
#         f.write(f"Episode {i}: Reward = {reward}\n")