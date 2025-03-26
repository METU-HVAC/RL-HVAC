import gymnasium as gym
import torch
import numpy as np
import random
from algorithms.ppo.ppo import *  # Ensure this is the correct import path for your PPOAgent

# Initialize the environment
seed = 42  # Set seed for reproducibility
# Set seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
# Hyperparameters
MAX_TESTING_EPISODES = 100  # Number of episodes for evaluation
MAX_EPISODE_STEPS = 1000  # Maximum steps per episode
#MODEL_PATH = "path_to_your_saved_model.pt"  # Replace with the path to your saved PPO model

MAX_TRAINING_STEPS = 50000
# Initialize the environment
env = gym.make("MountainCarContinuous-v0")
# If using environments that rely on

# Get the number of actions and observations
n_actions = 1
n_observations = env.observation_space.shape[0]

# Load the trained agent
agent = PPOAgent(device,n_actions, n_observations, MAX_EPISODE_STEPS,1e-3)  # Adjust hyperparameters as necessary
#agent.load(MODEL_PATH)  # Ensure `PPOAgent` has a `load` method for loading trained models
UPDATE_TIMESTEP = 4000
# Evaluation loop
total_rewards = []
total_timestep = 0
for episode in range(MAX_TESTING_EPISODES):
    obs, _ = env.reset(seed=seed + episode)  # Reset environment and set seed for reproducibility
    episode_reward = 0
    
    for step in range(MAX_EPISODE_STEPS):
        obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)  # Convert observation to tensor
        with torch.no_grad():
            action,value = agent.get_action_and_value(obs_tensor)  # Get action from the agent
        #action = action.item()  # Convert tensor to scalar for environment interaction
        # saving reward and is_terminals
        
        # Step the environment
        obs, reward, terminated, truncated, _ = env.step(action)

        agent.buffer.rewards.append(reward)
        agent.buffer.is_terminals.append(terminated or truncated)
        total_timestep += 1
        episode_reward += reward
        if total_timestep % UPDATE_TIMESTEP == 0:
            agent.update()
        if terminated or truncated:
            break

    total_rewards.append(episode_reward)
    print(f"Episode {episode + 1}/{MAX_TESTING_EPISODES} - Reward: {episode_reward}")

# Close the environment
env.close()

# Print results
average_reward = np.mean(total_rewards)
std_reward = np.std(total_rewards)
print(f"Average Reward over {MAX_TESTING_EPISODES} episodes: {average_reward}")
print(f"Standard Deviation of Rewards: {std_reward}")

# Plot the results
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(range(1, MAX_TESTING_EPISODES + 1), total_rewards, marker='o', label='Episode Reward')
plt.axhline(y=average_reward, color='r', linestyle='--', label='Average Reward')
plt.title('PPO Agent Testing Performance')
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.legend()
plt.grid()
plt.show()
