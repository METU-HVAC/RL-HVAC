import torch
import numpy as np

class RolloutBuffer:
    def __init__(self, num_steps, num_envs, obs_shape, act_shape, device):
        """
        Initializes the Sequential Replay Buffer.

        Args:
            num_steps (int): Total number of steps for each environment per buffer cycle.
            num_envs (int): Number of environments being simulated in parallel.
            obs_shape (tuple): Shape of the observation (e.g., (state_dim,)).
            act_shape (tuple): Shape of the action (e.g., (action_dim,)).
            device (torch.device): Device to store the tensors (e.g., CPU or CUDA).
        """
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.device = device

        # Initialize buffers for storing experience tuples
        self.obs_buf = torch.zeros((num_steps, num_envs, *obs_shape), dtype=torch.float32, device=self.device)
        self.next_obs_buf = torch.zeros_like(self.obs_buf, device=self.device)
        self.acts_buf = torch.zeros((num_steps, num_envs, *act_shape), dtype=torch.int64, device=self.device)
        self.rews_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.dones_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)
        self.logprobs_buf = torch.zeros((num_steps, num_envs), dtype=torch.float32, device=self.device)

        self.ptr = 0  # Pointer for where the next experience will be stored

    def add(self, obs, next_obs, actions, rewards, dones, logprobs):
        """
        Adds experience to the buffer.

        Args:
            obs (Tensor): Observation from the current timestep.
            next_obs (Tensor): Observation from the next timestep.
            actions (Tensor): Action taken at the current timestep.
            rewards (Tensor): Reward received after the action.
            dones (Tensor): Done flag indicating if episode has finished.
            logprobs (Tensor): Log probability of the taken action.
        """
        # Store the experience
        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = actions
        self.rews_buf[self.ptr] = rewards
        self.dones_buf[self.ptr] = dones
        self.logprobs_buf[self.ptr] = logprobs

        # Increment the pointer and wrap it around if necessary
        self.ptr = (self.ptr + 1) % self.num_steps

    def sample(self, batch_size):
        """
        Samples a batch of experiences from the buffer.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            dict: Batch of experience tuples (states, next_states, actions, rewards, etc.)
        """
        indices = np.random.randint(0, self.num_steps, size=batch_size)
        return {
            'obs': self.obs_buf[indices],
            'next_obs': self.next_obs_buf[indices],
            'actions': self.acts_buf[indices],
            'rewards': self.rews_buf[indices],
            'dones': self.dones_buf[indices],
            'logprobs': self.logprobs_buf[indices],
        }

    def get_state(self):
        """
        Returns the current state of the replay buffer.

        Returns:
            dict: Current buffer state containing all experience data.
        """
        return {
            'obs_buf': self.obs_buf,
            'next_obs_buf': self.next_obs_buf,
            'acts_buf': self.acts_buf,
            'rews_buf': self.rews_buf,
            'dones_buf': self.dones_buf,
            'logprobs_buf': self.logprobs_buf,
            'ptr': self.ptr,
        }

    def set_state(self, state):
        """
        Restores the state of the replay buffer.

        Args:
            state (dict): The state to restore, typically from a checkpoint.
        """
        self.obs_buf = state['obs_buf']
        self.next_obs_buf = state['next_obs_buf']
        self.acts_buf = state['acts_buf']
        self.rews_buf = state['rews_buf']
        self.dones_buf = state['dones_buf']
        self.logprobs_buf = state['logprobs_buf']
        self.ptr = state['ptr']

    def __len__(self):
        """Returns the number of experiences stored."""
        return self.ptr
