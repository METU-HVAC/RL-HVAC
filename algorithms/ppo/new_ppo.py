import os
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import math

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]
        
        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    T.nn.init.orthogonal_(layer.weight, std)
    T.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, n_actions,input_dims,total_timesteps,batch_size,lr=0.0003,gamma=0.99,gae_lambda=0.95,clip_coef = 0.2
                 ,vf_coef = 0.5,ent_coef = 0.01,max_grad_norm = 0.5):
        super(Agent, self).__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(*input_dims, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1), std=1.0),
        )
        self.actor = nn.Sequential(
            layer_init(nn.Linear(*input_dims, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, 1024)),
            nn.Tanh(),
            layer_init(nn.Linear(1024, n_actions), std=0.01),
        )
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.lr = lr
        self.total_timesteps = total_timesteps
        self.optimizer =optim.Adam(self.parameters(), lr=lr,eps=1e-5)
        self.memory = PPOMemory(batch_size)
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_coef = clip_coef
        self.vf_coef = vf_coef
        self.ent_coef = ent_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = 4
        self.clip_value_loss = True
        self.norm_adv = True
        self.to(self.device)
        
    def anneal_lr(self,timestep):
        frac = 1.0 - (timestep - 1.0) / self.total_timesteps
        lrnow = frac * self.lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lrnow   
        
    def get_value(self, observation):
        state = T.tensor([observation], dtype=T.float).to(self.device)
        return self.critic(state)
    
    def get_prob_and_entrophy(self, observation,action):
        if not isinstance(observation, T.Tensor):
            state = T.tensor([observation], dtype=T.float).to(self.device)
        else:
            state = observation.to(self.device)  # Ensure it's on the correct device

        logits = self.actor(state)
        logits= T.nn.functional.log_softmax(logits, dim=-1)
        probs = Categorical(logits=logits)
        
        return probs.log_prob(action), probs.entropy(), self.critic(state)
    def get_action_and_value(self, observation, action=None):
    # Check if observation is already a tensor
        if not isinstance(observation, T.Tensor):
            state = T.tensor([observation], dtype=T.float).to(self.device)
        else:
            state = observation.to(self.device)  # Ensure it's on the correct device

        logits = self.actor(state)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        return action.item(), probs.log_prob(action).item(), probs.entropy().item(), self.critic(state).item()
        
    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)
    def learn(self,is_done):
        for _ in range(1):
            state_arr, action_arr, old_prob_arr, vals_arr,\
                reward_arr, dones_arr, batches = self.memory.generate_batches()
                
            # Tensorize the values
            values = T.tensor(vals_arr, dtype=T.float32).to(self.device)
            rewards = T.tensor(reward_arr, dtype=T.float32).to(self.device)
            dones = T.tensor(dones_arr, dtype=T.float32).to(self.device)    
            
            
            # Calculate advantages using GAE
            advantages = T.zeros(len(rewards), dtype=T.float32).to(self.device)
            next_value = self.critic(T.tensor([state_arr[-1]], dtype=T.float).to(self.device))
            last_gaelam = 0
            with T.no_grad():
                for t in reversed(range(len(rewards))):
                    if t == len(rewards) - 1:
                        next_non_terminal = 1.0 - is_done # Last done is the last state
                        next_values = next_value  # Terminal state
                        
                    else:
                        next_non_terminal = 1.0 - dones[t + 1]
                        next_values = values[t + 1]
                        
                    
                    delta = rewards[t] + self.gamma * next_values* next_non_terminal - values[t]
                    last_gaelam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gaelam
                    advantages[t] = last_gaelam
                returns = advantages + values
            
            #First detach value,advantages and returns
            #Then convert them to tensor later
            
            adv_arr = advantages.detach().cpu().numpy()
            ret_arr = returns.detach().cpu().numpy()
            value_arr = values.detach().cpu().numpy()
            
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(self.device)
                old_probs = T.tensor(old_prob_arr[batch]).to(self.device)
                actions = T.tensor(action_arr[batch]).to(self.device)
                
                batch_advantages = T.tensor(adv_arr[batch], dtype=T.float).to(self.device)
                batch_returns = T.tensor(ret_arr[batch], dtype=T.float).to(self.device)
                batch_values = T.tensor(value_arr[batch], dtype=T.float).to(self.device)
                
                newlogprob, entropy, newvalue = self.get_prob_and_entrophy(states,actions)
                
                logratio = newlogprob - old_probs
                ratio = logratio.exp()
                
                # with T.no_grad():
                #     old_approx_kl = -logratio.mean().item()
                #     approx_kl = ((ratio - 1) - logratio).mean().item()
                if self.norm_adv:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
                
                
                # Policy loss
                pg_loss1 = -batch_advantages * ratio
                pg_loss2 = -batch_advantages * T.clamp(ratio, 1 - self.clip_coef, 1 + self.clip_coef)
                pg_loss = T.max(pg_loss1, pg_loss2).mean()
                
                #Value loss
                newvalue = newvalue.view(-1)
                if self.clip_value_loss:
                    v_loss_unclipped = (batch_returns - newvalue).pow(2)
                    v_clipped = batch_values + T.clamp(newvalue - batch_values, -self.clip_coef, self.clip_coef)
                    v_loss_clipped  = (batch_returns - v_clipped).pow(2)
                    v_loss_max = T.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * (batch_returns - newvalue).pow(2).mean()
                    
                #entrhopy loss
                
                entropy_loss = entropy.mean()
                loss = pg_loss - self.ent_coef * entropy_loss + v_loss * self.vf_coef
                #loss = pg_loss
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), self.max_grad_norm)
                self.optimizer.step()
            
        self.memory.clear_memory()    
        return loss.item()