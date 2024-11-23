
import os
import shutil
from datetime import datetime
import torch
#Before starting, clear any directories that may have been created in previous runs
#They start with Eplus-env-* and are created by EnergyPlus
def remove_previous_run_logs():
    for root, dirs, files in os.walk(".", topdown=False):
        for name in dirs:
            if name.startswith("Eplus-env-"):
                dir_path = os.path.join(root, name)
                shutil.rmtree(dir_path)

# Function to save the model
def save_best_model(agent, episode, val_reward, model_dir):
    """
    Save the model if the current validation reward is the best so far.
    
    Args:
    - agent: The agent object containing the policy network.
    - episode: The current episode number.
    - val_reward: The average validation reward for the current episode.
    - model_dir: The directory where the models are saved.
    """
    # Initialize best validation reward if not already initialized
    if not hasattr(save_best_model, "best_val_reward"):
        save_best_model.best_val_reward = float('-inf')

    # Check if the current validation reward is the best so far
    if val_reward > save_best_model.best_val_reward:
        save_best_model.best_val_reward = val_reward

        # Save the model
        current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        model_path = os.path.join(model_dir, f"best_model_ep{episode}_{current_time}.pt")
        torch.save(agent.policy_net.state_dict(), model_path)