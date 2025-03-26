import gymnasium as gym
import torch
import numpy as np
from algorithms.dqn.dqn import *
import matplotlib.pyplot as plt

seed = 42  # Set seed for reproducibility
# Set seeds for reproducibility
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
MAX_TRAINING_STEPS = 50000
# Initialize the environment
env = gym.make('CartPole-v1')
# If using environments that rely on `action_space.sample()`
# Get the number of actions and observations
n_actions = env.action_space.n
n_observations = env.observation_space.shape[0]
agent = DQNAgent(n_observations, n_actions,10000)


# Hyperparameters
target_reward = 400  # The reward goal for solving the environment
log_interval = 10    # Interval for calculating and printing the average reward
reward_history = []
total_steps = 0
episode = 0
while total_steps < MAX_TRAINING_STEPS:
    state,info = env.reset()
    state = torch.tensor([state], dtype=torch.float32, device=device)
    total_reward = 0
    done = False
    episode = episode + 1
    for t in range(1, 1000):  # Limit to 1000 steps per episode
        # Select and perform an action
        total_steps += 1
        action = agent.select_action(state)
        next_state, reward, terminated, truncated,_ = env.step(action.item())
        total_reward += reward
        done = terminated or truncated
        # If the episode is done, set next_state to None
        next_state = None if done else torch.tensor([next_state], dtype=torch.float32, device=device)
        
        # Store the transition in memory
        reward = torch.tensor([reward], device=device)
        agent.store_transition(state, action, next_state, reward)
        
        # Move to the next state
        state = next_state
        
        # Perform optimization step
        agent.optimize_model()
        
        if done:
            break
    reward_history.append(total_reward)

    if (episode ) % log_interval == 0:
        avg_reward = np.mean(reward_history[-log_interval:])
        print(f"Episode {episode}, Average Reward (last {log_interval} episodes): {avg_reward}")
        # Check if the environment is solved
        if avg_reward >= target_reward:
            print(f"Solved in episode {episode}!")
            break
# Plot the reward history
plt.plot(np.convolve(reward_history, np.ones(log_interval)/log_interval, mode='valid'))
plt.title("Average Reward per Episode")
plt.xlabel("Episode")
plt.ylabel("Average Reward")
plt.show()

env.close()
# Initialize the environment
render_env = gym.make('CartPole-v1',render_mode="human")
# Render the environment using the trained model
state,info = render_env.reset(seed=seed)
state = torch.tensor([state], dtype=torch.float32, device=device)
render_env.render()

done = False
total_reward = 0

print("\nRendering a new environment with the trained agent...")

while not done:
    # Use the trained model to select an action
    action = agent.choose_greedy_action(state)
    
    # Perform the action in the environment
    next_state, reward, terminated,truncated, _ = render_env.step(action.item())
    total_reward += reward
    done = terminated or truncated
    # Move to the next state
    state = torch.tensor([next_state], dtype=torch.float32, device=device)
    
    # Render the environment
    render_env.render()

print(f"Total reward achieved by the trained agent: {total_reward}")
render_env.close()
