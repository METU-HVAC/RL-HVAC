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
from typing import Dict, List, Any
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


def train_gnn(
    snapshots: List[Dict[str, Any]],
    stats: Dict[str, Dict[str, float]],
    semantic_root: str,
    epochs: int = 50,
    batch_size: int = 32,
    lr: float = 1e-3,
    latent_dim: int = 32,
    device: str = 'cuda'
) -> GraphEncoder:
    """Train the GNN encoder on collected snapshots."""
    
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    # Load ontology and graph structure
    ontology_path = os.path.join(semantic_root, "ontology", "building_ontology.ttl")
    mapping_path = os.path.join(semantic_root, "ontology", "mappings.yaml")
    
    ctx = create_ontology_context(ontology_path)
    mapping = load_mapping(mapping_path)
    graph = build_graph_structure(ctx, mapping)
    
    edge_index = torch.tensor(graph.edge_index, dtype=torch.long, device=device)
    
    # Initialize model
    in_dim = get_feature_dim(graph)
    hidden_dim = 64
    model = GraphEncoder(in_dim, hidden_dim, latent_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    # Training loop
    num_samples = len(snapshots)
    indices = list(range(num_samples))
    
    losses_history = []
    
    for epoch in range(epochs):
        random.shuffle(indices)
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
            batch_loss.backward()
            optimizer.step()
            
            epoch_loss += batch_loss.item()
            num_batches += 1
        
        avg_loss = epoch_loss / num_batches
        losses_history.append(avg_loss)
        
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.6f}")
    
    return model, losses_history


def main():
    parser = argparse.ArgumentParser(description="Train GNN Encoder with PMV prediction")
    parser.add_argument("--env-id", type=str, default="A403mediumfanger")
    parser.add_argument("--season", type=str, default="hot", choices=["hot", "cool", "mixed"])
    parser.add_argument("--chunks", type=int, default=5, help="Number of chunks to collect data from")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--latent-dim", type=int, default=32, help="Latent vector size for GraphEncoder")
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
    
    # Train model
    print("\n🧠 Phase 3: Training GNN encoder...")
    model, losses = train_gnn(
        snapshots, stats, semantic_root,
        epochs=args.epochs, lr=args.lr, latent_dim=args.latent_dim
    )
    
    # Save model
    model_path = os.path.join(models_dir, "graph_encoder.pt")
    torch.save(model.state_dict(), model_path)
    print(f"\n✅ Saved trained model to {model_path}")

    config_path = os.path.join(models_dir, "graph_config.json")
    with open(config_path, "w") as f:
        json.dump(
            {
                "env_id": args.env_id,
                "season": args.season,
                "latent_dim": args.latent_dim,
                "epochs": args.epochs,
                "lr": args.lr,
                "chunks": args.chunks,
            },
            f,
            indent=2,
        )
    print(f"Saved graph config to {config_path}")
    
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Final loss: {losses[-1]:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
