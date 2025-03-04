import os
import torch as T
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.distributions.normal import Normal
import numpy as np

class DiscreteSACCriticNetwork(nn.Module):
    def __init__(self, beta, n_observations, n_actions ):
        super(DiscreteSACCriticNetwork, self).__init__()

        self.fc1 = nn.Linear(*n_observations, 256)
        self.fc2 = nn.Linear(256, 128)
        self.q = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.q(x)  # No activation (raw Q-values)
        return q_values  # Returns a vector of Q-values (one per action)
    
class DiscreteSACValueNetwork(nn.Module):
    def __init__(self, beta, n_observations):
        super(DiscreteSACValueNetwork, self).__init__()

        self.fc1 = nn.Linear(*n_observations, 256)
        self.fc2 = nn.Linear(256, 128)
        self.v = nn.Linear(128, 1)

        self.optimizer = optim.Adam(self.parameters(), lr=beta)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        state_value = self.fc1(state)
        state_value = F.relu(state_value)
        state_value = self.fc2(state_value)
        state_value = F.relu(state_value)

        v = self.v(state_value)

        return v

class DiscreteSACActorNetwork(nn.Module):
    def __init__(self, alpha, n_observations,n_actions):
        super(DiscreteSACActorNetwork, self).__init__()

        self.fc1 = nn.Linear(*n_observations, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')

        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_logits = self.fc3(x)  # No activation (logits)
        return action_logits  # Will be passed through softmax externally
