import torch
import torch.nn as nn
import torch.nn.functional as F

class DQN(nn.Module):
    def __init__(self, n_observations, n_actions, layer_sizes=[128, 64]):
        super(DQN, self).__init__()
        self.layers = nn.ModuleList()
        
        # First layer: input to first hidden layer
        self.layers.append(nn.Linear(n_observations, layer_sizes[0]))
        
        # Hidden layers
        for i in range(len(layer_sizes) - 1):
            self.layers.append(nn.Linear(layer_sizes[i], layer_sizes[i+1]))
        
        # Output layer
        self.output_layer = nn.Linear(layer_sizes[-1], n_actions)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)