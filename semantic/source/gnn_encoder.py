import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

class GraphConvLayer(nn.Module):
    """
    Simple Graph Convolution Layer (GCN style).
    Implements: X' = D^-0.5 A D^-0.5 X W
    """
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_features))

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x (torch.Tensor): Node features (num_nodes, in_features).
            edge_index (torch.Tensor): Edge list (2, num_edges).
            
        Returns:
            torch.Tensor: Updated node features (num_nodes, out_features).
        """
        num_nodes = x.size(0)
        
        # Create dense adjacency matrix from edge_index (for simplicity)
        # Note: For large graphs, sparse implementation is better.
        adj = torch.zeros((num_nodes, num_nodes), device=x.device)
        if edge_weight is None:
            adj[edge_index[0], edge_index[1]] = 1.0
        else:
            adj[edge_index[0], edge_index[1]] = edge_weight
        
        # Add self-loops
        adj = adj + torch.eye(num_nodes, device=x.device)
        
        # Compute Degree matrix D
        degree = adj.sum(dim=1)
        
        # Compute D^-0.5
        d_inv_sqrt = torch.pow(degree, -0.5)
        d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
        
        # Symmetric normalized adjacency: D^-0.5 A D^-0.5
        adj_norm = d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
        
        # Graph convolution: A_norm * X * W
        out = self.linear(x)
        out = adj_norm @ out + self.bias
        
        return out

class GraphEncoder(nn.Module):
    """
    Graph Autoencoder-like model that encodes the building state into a room latent vector.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_edges: int = 0,
        learn_edge_weights: bool = False
    ):
        """
        Initialize the GraphEncoder.

        Args:
            in_dim (int): Input feature dimension per node.
            hidden_dim (int): Hidden dimension size for GNN layers.
            latent_dim (int): Size of the final latent vector z.
        """
        super().__init__()
        
        # Encoder: 3 Graph Convolution Layers
        self.conv1 = GraphConvLayer(in_dim, hidden_dim)
        self.conv2 = GraphConvLayer(hidden_dim, hidden_dim)
        self.conv3 = GraphConvLayer(hidden_dim, hidden_dim)
        self.learn_edge_weights = learn_edge_weights
        self.num_edges = num_edges
        if self.learn_edge_weights and self.num_edges > 0:
            # Static, globally learned edge gates in logit space.
            self.edge_logits = nn.Parameter(torch.zeros(self.num_edges))
        else:
            self.register_parameter("edge_logits", None)
        
        # Latent projection: hidden_dim -> latent_dim
        self.to_latent = nn.Linear(hidden_dim, latent_dim)

        # Dropout for regularization
        self.dropout = nn.Dropout(p=0.1)
        
        # Shared Decoder
        self.decoder_shared = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )
        
        # Decoder Heads (MLPs)
        # Predicting normalized values (typically 0-1 or standardized)
        
        self.head_air_temperature = nn.Linear(64, 1)
        self.head_air_humidity = nn.Linear(64, 1)
        self.head_air_co2 = nn.Linear(64, 1)
        self.head_occupancy = nn.Linear(64, 1)
        self.head_outdoor_temperature = nn.Linear(64, 1)
        self.head_outdoor_humidity = nn.Linear(64, 1)
        self.head_window_fan_energy = nn.Linear(64, 1)
        self.head_total_electricity_HVAC = nn.Linear(64, 1)
        self.head_pmv = nn.Linear(64, 1)  # PMV prediction for comfort

    def encode(self, x: torch.Tensor, edge_index: torch.Tensor, room_index: int) -> torch.Tensor:
        """
        Encode the graph into a latent vector representing the room.

        Args:
            x (torch.Tensor): Node features (num_nodes, in_dim).
            edge_index (torch.Tensor): Graph connectivity (2, num_edges).
            room_index (int): Index of the room node to extract embedding from.

        Returns:
            torch.Tensor: Latent vector z of shape (latent_dim,).
        """
        edge_weight = None
        if self.edge_logits is not None:
            edge_weight = torch.sigmoid(self.edge_logits)

        # Layer 1
        h = self.conv1(x, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Layer 2
        h = self.conv2(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.dropout(h)

        # Layer 3
        h = self.conv3(h, edge_index, edge_weight=edge_weight)
        h = F.relu(h)
        h = self.dropout(h)
        
        # Extract room node embedding
        room_embedding = h[room_index]
        
        # Project to latent space
        z = self.to_latent(room_embedding)
        
        return z

    def decode(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Decode the latent vector into specific predictions.

        Args:
            z (torch.Tensor): Latent vector (latent_dim,).

        Returns:
            dict: Predictions for specific sensors.
        """
        shared_out = self.decoder_shared(z)

        preds = {
            "pred_air_temperature":        self.head_air_temperature(shared_out),
            "pred_air_humidity":           self.head_air_humidity(shared_out),
            "pred_air_co2":                self.head_air_co2(shared_out),
            "pred_occupancy":              self.head_occupancy(shared_out),
            "pred_outdoor_temperature":    self.head_outdoor_temperature(shared_out),
            "pred_outdoor_humidity":       self.head_outdoor_humidity(shared_out),
            "pred_window_fan_energy":      self.head_window_fan_energy(shared_out),
            "pred_total_electricity_HVAC": self.head_total_electricity_HVAC(shared_out),
            "pred_pmv":                    self.head_pmv(shared_out),
        }
        return preds

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, room_index: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass combining encoding and decoding.

        Args:
            x (torch.Tensor): Node features.
            edge_index (torch.Tensor): Edge list.
            room_index (int): Room node index.

        Returns:
            tuple: (z, predictions_dict)
        """
        z = self.encode(x, edge_index, room_index)
        preds = self.decode(z)
        return z, preds

    def get_edge_weights(self) -> torch.Tensor:
        """
        Return current learned edge weights in [0, 1].
        """
        if self.edge_logits is None:
            return torch.empty(0)
        return torch.sigmoid(self.edge_logits)
