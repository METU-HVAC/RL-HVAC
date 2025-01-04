import torch.nn as nn
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
from torch.distributions.categorical import Categorical
import random
from algorithms.ppo.rollout_buffer import RolloutBuffer
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    nn.init.orthogonal_(layer.weight, std)
    nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    def __init__(self, input_dim, action_dim):
        super(ActorCritic, self).__init__()
        # Actor
        self.actor = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, action_dim), std=0.01)
        )
        
        # Critic
        self.critic = nn.Sequential(
            layer_init(nn.Linear(input_dim, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 256)),
            nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0)
        )

    def forward(self, x):
        value = self.critic(x)
        logits = self.actor(x)
        return logits, value
    
class PPOAgent(nn.Module):
    def __init__(self, device, num_actions, num_states, num_steps, lr):
        super(PPOAgent,self).__init__()
        self.device = device
        self.num_steps = num_steps   
        self.num_states = num_states
        self.num_actions = num_actions
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.minibatch_size = 16
        self.K_epochs = 1
        self.adv_norm = True
        self.clip_param = 0.2
        self.clip_vloss = True
        self.entropy_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        self.buffer_capacity = 1000000
        #self.num_states = np.array(envs.single_observation_space.shape).prod()
        # Replay buffer
        self.replay_buffer = RolloutBuffer( num_steps=35040, 
                                num_envs=1, 
                                obs_shape=(self.num_states,),  # Example observation shape
                                act_shape=(self.num_actions,),  # Example action shape
                                device=self.device)

        self.actor_critic = ActorCritic(num_states, num_actions)
        self.optimizer = torch.optim.Adam([
                        {'params': self.actor_critic.actor.parameters(), 'lr': lr},
                        {'params': self.actor_critic.critic.parameters(), 'lr': lr}
                    ])
        self.to(self.device)

    def get_value(self,x):
        return self.critic(x) 
    
    def get_action_and_value(self, states):
        logits, value = self.actor_critic(states)
        dist = torch.distributions.Categorical(logits=logits)
        actions = dist.sample()
        log_probs = dist.log_prob(actions).clone().detach()  # Clone and detach
        value = value.clone().detach()  # Ensure value is detached too
        return actions, log_probs, value

    
    def save_checkpoint(self, checkpoint_path):
        """
        Save the model, optimizer, and replay buffer states.

        Args:
            checkpoint_dir (str): Directory to save the checkpoint.
            epoch (int): Current epoch of training.
            timestep (int): Current training timestep.
        """
        
        checkpoint = {
            'actor_critic_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            # Optionally include replay buffer if needed
            # 'replay_buffer': self.replay_buffer.get_state(),  # Add serialize method if replay buffer is large
        }
        
        torch.save(checkpoint, checkpoint_path)
        print(f"Checkpoint saved at {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path):
        """
        Load the model, optimizer, and replay buffer states.

        Args:
            checkpoint_path (str): Path to the checkpoint file.

        Returns:
            int, int: Restored epoch and timestep.
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['actor_critic_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # Optionally restore replay buffer if saved
        # self.replay_buffer.set_state(checkpoint["replay_buffer"])
        
        print(f"Checkpoint loaded from {checkpoint_path}")
    
    def load(self,checkpoint_path):
        self.load_state_dict(torch.load(checkpoint_path))

    def update(self):
        """
        Perform a PPO update using the replay buffer.

        Returns:
            policy_loss (float): Final policy loss after the update.
            value_loss (float): Final value loss after the update.
        """
        # Extract data from the replay buffer
        states = self.replay_buffer.obs_buf.view(-1, self.num_states).clone()
        next_states = self.replay_buffer.next_obs_buf.view(-1, self.num_states).clone()
        actions = self.replay_buffer.acts_buf.view(-1).clone()
        log_probs_old = self.replay_buffer.logprobs_buf.view(-1).clone()
        rewards = self.replay_buffer.rews_buf.view(-1).clone()
        dones = self.replay_buffer.dones_buf.view(-1).clone()

        # Compute values and next values
        with torch.no_grad():
            _, values = self.actor_critic(states)
            _, next_values = self.actor_critic(next_states)

            values = values.squeeze()
            next_values = next_values.squeeze()

            # GAE computation
            advantages = torch.zeros_like(rewards).to(self.device)
            advantages = advantages.detach()
            gae = 0
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + self.gamma * next_values[t] * (1 - dones[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones[t]) * gae
                advantages[t] = gae
            returns = advantages + values

            # Normalize advantages
            if self.adv_norm:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # PPO Update
        for _ in range(self.K_epochs):
            indices = torch.randperm(len(states))
            for start in range(0, len(states), self.minibatch_size):
                end = start + self.minibatch_size
                batch_indices = indices[start:end]

                # Sample mini-batch and detach tensors
                batch_states = states[batch_indices].detach()
                batch_actions = actions[batch_indices].detach()
                batch_log_probs_old = log_probs_old[batch_indices].detach()
                batch_advantages = advantages[batch_indices].detach()
                batch_returns = returns[batch_indices].detach()

                # Compute policy outputs
                logits, values = self.actor_critic(batch_states)
                dist = torch.distributions.Categorical(logits=logits)
                new_log_probs = dist.log_prob(batch_actions)
                entropy = dist.entropy().mean()

                # PPO surrogate loss
                ratio = torch.exp(new_log_probs - batch_log_probs_old)
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                values = values.squeeze()
                value_loss = nn.MSELoss()(values, batch_returns)
                if self.clip_vloss:
                    value_clipped = batch_returns + torch.clamp(values - batch_returns, -self.clip_param, self.clip_param)
                    value_loss_clipped = nn.MSELoss()(value_clipped, batch_returns)
                    value_loss = 0.5 * (value_loss + value_loss_clipped)

                # Total loss
                total_loss = policy_loss + self.vf_coef * value_loss - self.entropy_coef * entropy

                # Update actor-critic
                self.optimizer.zero_grad()
                total_loss.backward(retain_graph=True)  # Removed retain_graph=True
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)
                self.optimizer.step()

        return 0.5 * policy_loss + 0.5 * value_loss