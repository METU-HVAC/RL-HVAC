import os
import torch
import numpy as np
import json
from typing import Dict, Any, Tuple

from .ontology_loader import create_ontology_context, OntologyContext
from .state_vector_builder import load_mapping
from .graph_builder import build_graph_structure, GraphStructure
from .node_features import build_node_features, get_feature_dim
from .gnn_encoder import GraphEncoder

class SemanticStateProvider:
    """
    Helper class to provide semantic state representations from the graph encoder.
    Used to bridge the trained GNN with the RL environment.
    """

    def __init__(self, 
                 semantic_root: str, 
                 device: str = "cpu",
                 model_path: str = None,
                 mapping_path: str = None,
                 ontology_path: str = None,
                 stats_path: str = None,
                 graph_artifacts_path: str = None):
        """
        Initialize the SemanticStateProvider.

        Args:
            semantic_root (str): Path to the semantic package root directory.
            device (str): Device to run the model on ('cpu' or 'cuda').
            model_path (str, optional): Path to the pretrained model file.
            mapping_path (str, optional): Path to the mappings.yaml file.
            ontology_path (str, optional): Path to the building_ontology.ttl file.
        """
        self.device = torch.device(device)
        self.semantic_root = semantic_root

        # 1. Load Ontology and Mapping
        if ontology_path is None:
            ontology_path = os.path.join(semantic_root, "ontology", "building_ontology.ttl")
        if mapping_path is None:
            mapping_path = os.path.join(semantic_root, "ontology", "mappings.yaml")
        
        if not os.path.exists(ontology_path):
            raise FileNotFoundError(f"Ontology file not found at: {ontology_path}")
        if not os.path.exists(mapping_path):
            raise FileNotFoundError(f"Mapping file not found at: {mapping_path}")

        self.ctx = create_ontology_context(ontology_path)
        self.mapping = load_mapping(mapping_path)

        default_graph_artifacts_path = os.path.join(semantic_root, "models", "graph_artifacts.json")
        graph_artifacts_path = graph_artifacts_path or default_graph_artifacts_path
        self.graph_artifacts = {}
        if os.path.exists(graph_artifacts_path):
            with open(graph_artifacts_path, "r") as f:
                self.graph_artifacts = json.load(f)

        # 2. Build Graph Structure (respect saved topology/pruned edges when available)
        graph_topology = self.graph_artifacts.get("topology", "star")
        self.graph = build_graph_structure(self.ctx, self.mapping, topology=graph_topology)
        pruned_edge_index = self.graph_artifacts.get("pruned_edge_index")
        if pruned_edge_index:
            self.graph.edge_index = np.array(pruned_edge_index, dtype=np.int64)
        
        # Pre-compute edge index tensor
        self.edge_index = torch.tensor(self.graph.edge_index, dtype=torch.long).to(self.device)

        # 3. Load Model and Stats
        in_dim = get_feature_dim(self.graph)
        
        # Default dimensions; may be overridden from checkpoint metadata.
        self.hidden_dim = 64
        self.latent_dim = 32
        
        default_model_path = os.path.join(semantic_root, "models", "graph_encoder.pt")
        default_stats_path = os.path.join(semantic_root, "models", "target_stats.json")
        using_custom_model = model_path is not None
        model_path = model_path or default_model_path
        stats_path = stats_path or default_stats_path
        self.loaded_model_path = model_path
        
        if os.path.exists(model_path):
            try:
                state_dict = torch.load(model_path, map_location=self.device)
                self.hidden_dim, self.latent_dim = self._infer_dims_from_state_dict(state_dict)
                learned_edge_logits = state_dict.get("edge_logits")
                can_use_learned_edges = (
                    learned_edge_logits is not None and
                    learned_edge_logits.shape[0] == self.edge_index.shape[1]
                )
                self.model = GraphEncoder(
                    in_dim,
                    self.hidden_dim,
                    self.latent_dim,
                    num_edges=self.edge_index.shape[1],
                    learn_edge_weights=can_use_learned_edges
                )
                self.model.load_state_dict(state_dict, strict=can_use_learned_edges)
                if learned_edge_logits is not None and not can_use_learned_edges:
                    print("Warning: edge_logits shape does not match loaded graph edges. Using fixed adjacency for inference.")
            except RuntimeError as e:
                print(f"Warning: Failed to load model weights directly: {e}")
                print("Tip: Ensure the model architecture (hidden_dim, latent_dim) matches the checkpoint.")
                raise e
        else:
            self.model = GraphEncoder(
                in_dim,
                self.hidden_dim,
                self.latent_dim,
                num_edges=self.edge_index.shape[1],
                learn_edge_weights=False
            )
            if using_custom_model:
                raise FileNotFoundError(f"Model checkpoint not found at {model_path}.")
            print(f"Warning: Model checkpoint not found at {model_path}. Initializing with random weights.")

        if os.path.exists(stats_path):
            with open(stats_path, "r") as f:
                self.target_stats = json.load(f)
        else:
             # Warning instead of error, as stats might not be needed for pure inference if no denormalization is done
             print(f"Warning: Target stats file not found at {stats_path}.")
             self.target_stats = {}

        self.model.to(self.device)
        self.model.eval()
        
        # Map pred keys to data keys
        self.key_mapping = {
            "pred_air_temperature":        "air_temperature",
            "pred_air_humidity":           "air_humidity",
            "pred_air_co2":                "air_co2",
            "pred_occupancy":              "people_occupant",
            "pred_outdoor_temperature":    "outdoor_temperature",
            "pred_outdoor_humidity":       "outdoor_humidity",
            "pred_window_fan_energy":      "window_fan_energy",
            "pred_total_electricity_HVAC": "total_electricity_HVAC",
            "pred_pmv":                    "pmv",
        }

    @staticmethod
    def _infer_dims_from_state_dict(state_dict: Dict[str, torch.Tensor]) -> Tuple[int, int]:
        """
        Infer (hidden_dim, latent_dim) from a serialized GraphEncoder state_dict.
        """
        hidden_dim = int(state_dict["conv1.linear.weight"].shape[0])
        latent_dim = int(state_dict["to_latent.weight"].shape[0])
        return hidden_dim, latent_dim

    def _forward_from_snapshot(self, snapshot: Dict[str, float]) -> Tuple[np.ndarray, Dict[str, torch.Tensor]]:
        """
        Internal helper to run the model from a snapshot.
        """
        x_np = build_node_features(
            self.graph, 
            snapshot, 
            mapping=self.mapping, 
            ctx=self.ctx
        )
        
        x = torch.tensor(x_np, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            z, preds = self.model(x, self.edge_index, self.graph.room_index)
            
        return z.cpu().numpy(), preds

    def get_state_from_snapshot(self, snapshot: Dict[str, float]) -> np.ndarray:
        """
        Given a snapshot dict, return the latent vector z_t.
        """
        z, _ = self._forward_from_snapshot(snapshot)
        return z

    def get_predicted_observables_from_snapshot(self, snapshot: Dict[str, float]) -> Dict[str, float]:
        """
        Given a snapshot dict, return the predicted observables (denormalized).
        """
        _, preds = self._forward_from_snapshot(snapshot)
        
        result = {}
        for pred_key, data_key in self.key_mapping.items():
            if pred_key in preds:
                val_norm = preds[pred_key].item()
                
                # De-normalize
                if data_key in self.target_stats:
                    mean = self.target_stats[data_key]["mean"]
                    std = self.target_stats[data_key]["std"]
                    val = val_norm * std + mean
                else:
                    val = val_norm
                
                result[data_key] = float(val)

        return result
