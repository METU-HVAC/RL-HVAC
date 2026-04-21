#!/usr/bin/env python3
"""
GNN Encoder Training Script

Trains the GraphEncoder model to predict building observables (including PMV) 
from the ontology-based graph representation.

Usage:
    cd /storage/Master/ParallelRL/RL-HVAC
    python semantic/train_gnn.py --episodes 5 --season hot
"""

import argparse
import os
import sys
import json
import random
from datetime import datetime
from typing import Dict, List, Any, Tuple
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Add paths
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sinergym
from sinergym.utils.constants import YEAR

from semantic.source.ontology_loader import create_ontology_context
from semantic.source.state_vector_builder import load_mapping
from semantic.source.graph_builder import build_graph_structure
from semantic.source.node_features import build_node_features, get_feature_dim
from semantic.source.gnn_encoder import GraphEncoder

from environments.reward import CO2andPMVReward
from environments.environment import create_environment
from utils.dataset import generate_chunks, balanced_month_sample


# Observation variables from environment
OBSERVATION_VARIABLES = [
    'month', 'day_of_month', 'hour',
    'outdoor_temperature', 'outdoor_humidity',
    'htg_setpoint', 'clg_setpoint', 'air_temperature',
    'air_humidity', 'people_occupant', 'air_co2',
    'window_fan_energy', 'pmv', 'ppd', 'total_electricity_HVAC',
]

# Target variables for GNN prediction (normalized)
TARGET_KEYS = [
    'air_temperature', 'air_humidity', 'air_co2', 'people_occupant',
    'outdoor_temperature', 'outdoor_humidity', 
    'window_fan_energy', 'total_electricity_HVAC', 'pmv'
]

# Prediction key mapping
PRED_TO_TARGET = {
    'pred_air_temperature': 'air_temperature',
    'pred_air_humidity': 'air_humidity',
    'pred_air_co2': 'air_co2',
    'pred_occupancy': 'people_occupant',
    'pred_outdoor_temperature': 'outdoor_temperature',
    'pred_outdoor_humidity': 'outdoor_humidity',
    'pred_window_fan_energy': 'window_fan_energy',
    'pred_total_electricity_HVAC': 'total_electricity_HVAC',
    'pred_pmv': 'pmv',
}


def collect_data_from_env(
    env_id: str,
    chunks: List,
    season: str,
    timesteps_per_hour: int = 4,
    max_steps_per_chunk: int = 768
) -> List[Dict[str, Any]]:
    """
    Collect snapshots from the environment for GNN training.
    
    Returns:
        List of snapshot dictionaries with all observation variables.
    """
    snapshots = []
    
    reward_config = {
        'pmv_variables': ['pmv'],
        'co2_variable': 'air_co2',
        'energy_variables': ['total_electricity_HVAC', 'window_fan_energy'],
        'ac_energy_weight': 0.6,
        'fan_energy_weight': 0.5,
        'co2_weight': 0.5,
        'pmv_weight': 0.4,
        'lambda_energy': 1/1_600_000,
        'lambda_pmv': 1.0,
        'lambda_co2': 1.0,
        'co2_threshold': 800,
    }
    
    for chunk in tqdm(chunks, desc="Collecting data"):
        start_date, end_date, chunk_season = chunk
        
        env = create_environment(
            env_id, start_date, end_date, season,
            CO2andPMVReward, episode_type="Validation",
            timesteps_per_hour=timesteps_per_hour,
            reward_kwargs=reward_config
        )
        
        state, info = env.reset()
        step = 0
        
        while step < max_steps_per_chunk:
            # Build snapshot from current state
            snapshot = dict(zip(OBSERVATION_VARIABLES, state))
            snapshots.append(snapshot)
            
            # Random action to explore state space
            action = random.randint(0, 39)
            next_state, _, truncated, terminated, _ = env.step(action)
            
            if truncated or terminated:
                state, _ = env.reset()
            else:
                state = next_state
            
            step += 1
        
        env.close()
    
    return snapshots


def compute_target_stats(snapshots: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Compute mean and std for each target variable."""
    stats = {}
    
    for key in TARGET_KEYS:
        values = [s[key] for s in snapshots if key in s]
        if values:
            stats[key] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)) if np.std(values) > 1e-6 else 1.0
            }
    
    return stats


def normalize_targets(snapshot: Dict[str, Any], stats: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    """Normalize target values using computed stats."""
    normalized = {}
    for key in TARGET_KEYS:
        if key in snapshot and key in stats:
            val = snapshot[key]
            mean = stats[key]['mean']
            std = stats[key]['std']
            normalized[key] = (val - mean) / std
    return normalized


def prune_edges_by_threshold(
    edge_index: np.ndarray,
    edge_weights: np.ndarray,
    threshold: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prune directed edges using a weight threshold.
    """
    if edge_index.size == 0 or edge_weights.size == 0:
        return edge_index, edge_weights

    mask = edge_weights >= threshold
    pruned_edge_index = edge_index[:, mask]
    pruned_edge_weights = edge_weights[mask]
    return pruned_edge_index, pruned_edge_weights


def build_graph_artifacts(
    graph,
    model: GraphEncoder,
    prune_threshold: float
) -> Dict[str, Any]:
    """
    Build graph metadata including learned edge weights and pruned graph.
    """
    artifacts: Dict[str, Any] = {
        "topology": graph.topology,
        "num_nodes": len(graph.node_uris),
        "num_edges": int(graph.edge_index.shape[1]) if graph.edge_index.size > 0 else 0,
        "edge_index": graph.edge_index.tolist(),
        "room_index": graph.room_index,
        "node_uris": graph.node_uris,
        "node_types": graph.node_types,
        "prune_threshold": prune_threshold,
    }

    learned_weights = model.get_edge_weights().detach().cpu().numpy()
    if learned_weights.size > 0:
        pruned_edge_index, pruned_edge_weights = prune_edges_by_threshold(
            graph.edge_index,
            learned_weights,
            prune_threshold
        )
        artifacts["edge_weights"] = learned_weights.tolist()
        artifacts["mean_edge_weight"] = float(learned_weights.mean())
        artifacts["pruned_edge_index"] = pruned_edge_index.tolist()
        artifacts["pruned_edge_weights"] = pruned_edge_weights.tolist()
        artifacts["num_pruned_edges"] = int(pruned_edge_index.shape[1]) if pruned_edge_index.size > 0 else 0
    else:
        artifacts["edge_weights"] = []
        artifacts["mean_edge_weight"] = 0.0
        artifacts["pruned_edge_index"] = graph.edge_index.tolist()
        artifacts["pruned_edge_weights"] = []
        artifacts["num_pruned_edges"] = int(graph.edge_index.shape[1]) if graph.edge_index.size > 0 else 0

    return artifacts


def train_gnn(
    snapshots: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, float]],
    semantic_root: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 32,
    device: str = 'cuda',
    topology: str = "star",
    learn_edge_weights: bool = False,
    sparse_lambda: float = 0.0,
    seed: int = 42
) -> Tuple[GraphEncoder, List[float], Any]:
    """Train the GNN encoder on collected snapshots."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load ontology and graph structure
    ontology_path = os.path.join(semantic_root, "ontology", "building_ontology.ttl")
    mapping_path = os.path.join(semantic_root, "ontology", "mappings.yaml")
    
    ctx = create_ontology_context(ontology_path)
    mapping = load_mapping(mapping_path)
    graph = build_graph_structure(ctx, mapping, topology=topology)
    
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    
    # Initialize model
    in_dim = get_feature_dim(graph)
    hidden_dim = 64
    model = GraphEncoder(
        in_dim,
        hidden_dim,
        latent_dim,
        num_edges=edge_index.shape[1],
        learn_edge_weights=learn_edge_weights
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    num_samples = len(snapshots)
    indices = list(range(num_samples))
    rng = random.Random(seed)
    
    losses_history = []
    
    for epoch in range(epochs):
        rng.shuffle(indices)
        epoch_loss = 0.0
        num_batches = 0
        
        model.train()
        
        for i in range(0, num_samples, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_loss = 0.0
            
            optimizer.zero_grad()
            
            for idx in batch_indices:
                snapshot = snapshots[idx]
                
                # Build node features
                x_np = build_node_features(graph, snapshot, mapping=mapping, ctx=ctx)
                x = torch.tensor(x_np, dtype=torch.float32, device=device)
                
                # Forward pass
                z, preds = model(x, edge_index, graph.room_index)
                
                # Compute loss for each prediction head
                normalized_targets = normalize_targets(snapshot, stats)
                
                for pred_key, target_key in PRED_TO_TARGET.items():
                    if pred_key in preds and target_key in normalized_targets:
                        target_val = torch.tensor(
                            [normalized_targets[target_key]], 
                            dtype=torch.float32, 
                            device=device
                        )
                        pred_val = preds[pred_key].squeeze()
                        batch_loss += criterion(pred_val, target_val.squeeze())
            
            batch_loss = batch_loss / len(batch_indices)
            if learn_edge_weights and model.edge_logits is not None and sparse_lambda > 0.0:
                batch_loss = batch_loss + (sparse_lambda * model.get_edge_weights().mean())
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    return model, losses_history, graph


def main():
    parser = argparse.ArgumentParser(description="Train GNN Encoder with PMV prediction")
    parser.add_argument("--env-id", type=str, default="A403mediumfanger")
    parser.add_argument("--season", type=str, default="hot", choices=["hot", "cool", "mixed"])
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks to collect data from")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent vector size for GraphEncoder")
    parser.add_argument("--topology", type=str, default="star", choices=["star", "fully_connected"])
    parser.add_argument("--learn-edge-weights", action="store_true", help="Learn static per-edge weights.")
    parser.add_argument("--sparse-lambda", type=float, default=1e-3, help="L1-like sparsity regularization weight.")
    parser.add_argument("--prune-threshold", type=float, default=0.1, help="Threshold for pruning learned edges.")
    parser.add_argument("--compare-topologies", action="store_true", help="Run matched star vs fully_connected comparison.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    # Set seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    semantic_root = "./semantic"
    models_dir = os.path.join(semantic_root, "models")
    os.makedirs(models_dir, exist_ok=True)
    
    print("=" * 60)
    print("GNN Encoder Training with PMV Prediction")
    print("=" * 60)
    
    # Generate chunks
    start_date = datetime(1997, 1, 1)
    all_chunks = generate_chunks(start_date, 8, 365, step_size=8, seasons=[args.season])
    
    # Use subset for training
    train_chunks = all_chunks[:args.chunks]
    print(f"Using {len(train_chunks)} chunks for data collection")
    
    # Collect data
    print("\n📊 Phase 1: Collecting data from environment...")
    snapshots = collect_data_from_env(
        args.env_id, train_chunks, args.season,
        max_steps_per_chunk=768
    )
    print(f"Collected {len(snapshots)} snapshots")
    
    # Compute statistics
    print("\n📈 Phase 2: Computing target statistics...")
    stats = compute_target_stats(snapshots)
    
    # Save stats
    stats_path = os.path.join(models_dir, "target_stats.json")
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)
    print(f"Saved target stats to {stats_path}")
    
    # Print PMV stats
    if 'pmv' in stats:
        print(f"PMV stats: mean={stats['pmv']['mean']:.4f}, std={stats['pmv']['std']:.4f}")
    
    print("\nPhase 3: Training GNN encoder...")

    run_summaries = []
    if args.compare_topologies:
        run_specs = [
            {"name": "star", "topology": "star", "learn_edge_weights": False},
            {"name": "fully_connected", "topology": "fully_connected", "learn_edge_weights": True},
        ]
    else:
        run_specs = [
            {
                "name": args.topology,
                "topology": args.topology,
                "learn_edge_weights": args.learn_edge_weights,
            }
        ]

    for spec in run_specs:
        run_name = spec["name"]
        print(f"\nTraining run: {run_name} (topology={spec['topology']}, learn_edge_weights={spec['learn_edge_weights']})")
        model, losses, graph = train_gnn(
            snapshots,
            stats,
            semantic_root,
            epochs=args.epochs,
            lr=args.lr,
            latent_dim=args.latent_dim,
            topology=spec["topology"],
            learn_edge_weights=spec["learn_edge_weights"],
            sparse_lambda=args.sparse_lambda,
            seed=args.seed,
        )

        model_filename = "graph_encoder.pt" if len(run_specs) == 1 else f"graph_encoder_{run_name}.pt"
        model_path = os.path.join(models_dir, model_filename)
        torch.save(model.state_dict(), model_path)
        print(f"Saved trained model to {model_path}")

        graph_artifacts = build_graph_artifacts(graph, model, args.prune_threshold)
        artifacts_filename = "graph_artifacts.json" if len(run_specs) == 1 else f"graph_artifacts_{run_name}.json"
        artifacts_path = os.path.join(models_dir, artifacts_filename)
        with open(artifacts_path, "w") as f:
            json.dump(graph_artifacts, f, indent=2)
        print(f"Saved graph artifacts to {artifacts_path}")

        run_summary = {
            "run_name": run_name,
            "topology": spec["topology"],
            "learn_edge_weights": spec["learn_edge_weights"],
            "final_loss": float(losses[-1]),
            "best_loss": float(min(losses)),
            "num_nodes": graph_artifacts["num_nodes"],
            "num_edges": graph_artifacts["num_edges"],
            "num_pruned_edges": graph_artifacts["num_pruned_edges"],
            "mean_edge_weight": graph_artifacts["mean_edge_weight"],
            "model_path": model_path,
            "graph_artifacts_path": artifacts_path,
        }
        run_summaries.append(run_summary)

    config_path = os.path.join(models_dir, "graph_config.json")
    config_payload = {
        "env_id": args.env_id,
        "season": args.season,
        "latent_dim": args.latent_dim,
        "epochs": args.epochs,
        "lr": args.lr,
        "chunks": args.chunks,
        "seed": args.seed,
        "sparse_lambda": args.sparse_lambda,
        "prune_threshold": args.prune_threshold,
        "compare_topologies": args.compare_topologies,
        "runs": run_summaries,
    }
    with open(config_path, "w") as f:
        json.dump(config_payload, f, indent=2)
    print(f"Saved graph config to {config_path}")

    if args.compare_topologies:
        comparison_path = os.path.join(models_dir, "topology_comparison.json")
        with open(comparison_path, "w") as f:
            json.dump({"runs": run_summaries}, f, indent=2)
        print(f"Saved topology comparison to {comparison_path}")

    print("\n" + "=" * 60)
    print("Training Complete!")
    for summary in run_summaries:
        print(
            f"{summary['run_name']}: final_loss={summary['final_loss']:.6f}, "
            f"edges={summary['num_edges']}, pruned_edges={summary['num_pruned_edges']}"
        )
    print("=" * 60)


if __name__ == "__main__":
    main()
