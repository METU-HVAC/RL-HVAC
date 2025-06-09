import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions,layer_sizes=[128,64]):
        super(DQN, self).__init__()
        self.layer_sizes = layer_sizes
        self.layer1 = nn.Linear(n_observations, layer_sizes[0])
        self.layer2 = nn.Linear(layer_sizes[0], layer_sizes[1])
        self.layer3 = nn.Linear(layer_sizes[1], n_actions)
        
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)