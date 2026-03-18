# Semantic GNN Comparison Tutorial

This tutorial shows how to compare RL agents trained with and without semantic GNN features.

---

## Quick Start

```bash
# Setup environment
conda activate hvac-rl
export PYTHONPATH=~/EnergyPlus-24.1.0:$PYTHONPATH
cd /storage/Master/ParallelRL/RL-HVAC
```

---

## Step 1: Train Models (3 Modes)

### Mode 1: RAW Model (Baseline)
```bash
python train_standalone.py \
  --episodes 10 \
  --season hot \
  --output-dir ./comparison
```
- **State size**: 13 dimensions
- **Input**: Standard observations only

### Mode 2: SEMANTIC CONCAT (Default)
```bash
python train_standalone.py \
  --episodes 10 \
  --season hot \
  --use-semantic \
  --output-dir ./comparison
```
- **State size**: 45 dimensions (13 base + 32 GNN)
- **Input**: `[base_observations, gnn_latent]`

### Mode 3: SEMANTIC LATENT-ONLY
```bash
python train_standalone.py \
  --episodes 10 \
  --season hot \
  --use-semantic \
  --semantic-mode latent \
  --output-dir ./comparison
```
- **State size**: 32 dimensions (GNN only)
- **Input**: GNN latent vector only (no raw observations)

---

## Semantic Modes Explained

| Mode | Flag | State Dim | Description |
|------|------|-----------|-------------|
| **Raw** | (none) | 13 | Standard observations |
| **Concat** | `--use-semantic` | 45 | Base obs + GNN latent |
| **Latent** | `--use-semantic --semantic-mode latent` | 32 | GNN latent only |

---

## Step 2: Evaluate Models

```bash
# Evaluate RAW model
python evaluate_standalone.py \
  --model-path ./comparison/*_raw_*/dqn_best.pth \
  --season hot

# Evaluate SEMANTIC CONCAT model
python evaluate_standalone.py \
  --model-path ./comparison/*_semantic_*/dqn_best.pth \
  --season hot \
  --use-semantic

# Evaluate SEMANTIC LATENT model
python evaluate_standalone.py \
  --model-path ./comparison/*_semantic_*/dqn_best.pth \
  --season hot \
  --use-semantic \
  --semantic-mode latent
```

> **Important**: Use the same `--semantic-mode` for evaluation as you did for training!

---

## Step 3: Compare Results

Results are saved to `results.json` in each evaluation folder:

```bash
cat ./eval_comparison/eval_*_raw_*/results.json
cat ./eval_comparison/eval_*_semantic_*/results.json
```

### Key Metrics to Compare

| Metric | Description | Better |
|--------|-------------|--------|
| `avg_reward` | Episode reward | Higher ↑ |
| `avg_power_kwh` | Energy used | Lower ↓ |
| `pmv_violation_pct` | Thermal discomfort % | Lower ↓ |
| `co2_violation_pct` | Poor air quality % | Lower ↓ |

---

## Step 4: Cross-Climate Generalization

Test if semantic features help with generalization:

```bash
# Train on HOT climate
python train_standalone.py --episodes 10 --season hot --output-dir ./generalization
python train_standalone.py --episodes 10 --season hot --use-semantic --output-dir ./generalization

# Evaluate on COOL climate (different from training)
python evaluate_standalone.py \
  --model-path ./generalization/*_raw_*/dqn_best.pth \
  --season cool

python evaluate_standalone.py \
  --model-path ./generalization/*_semantic_*/dqn_best.pth \
  --season cool \
  --use-semantic
```

---

## Example Results Comparison

After running both evaluations, you might see:

```
=== RAW Model ===
  Average Reward:      -250.86
  Average Power (kWh): 89.61
  PMV Violation (%):   24.57

=== SEMANTIC Model ===
  Average Reward:      -235.42  (+6% better)
  Average Power (kWh): 85.20   (5% less energy)
  PMV Violation (%):   22.10   (better comfort)
```

---

## Understanding the Semantic Module

The GNN encodes *building structure* into a 32-dim latent vector:

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Building  │────▶│    Room     │────▶│   Sensors   │
│  Ontology   │     │   A403      │     │ (Temp, CO2) │
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  GNN Latent │  z_t ∈ ℝ³²
                    │   Vector    │
                    └─────────────┘
                           │
                           ▼
        ┌──────────────────────────────────────┐
        │  Observation = [base_obs, z_t]       │
        │  (13 dims)      (32 dims) = 45 total │
        └──────────────────────────────────────┘
```

---

## Training Parameters Table

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `--episodes` | 10-20 | More for semantic (needs to learn graph) |
| `--season` | hot/cool/mixed/ankara | Start with one, then cross-test |
| `--learning-rate` | 3e-3 | Default works well |
| `--co2-weight` | 0.5 | Air quality importance |
| `--pmv-weight` | 0.4 | Thermal comfort importance |

---

## Files Reference

| File | Description |
|------|-------------|
| `train_standalone.py` | Training script |
| `evaluate_standalone.py` | Evaluation script |
| `semantic/` | GNN module |
| `semantic/ontology/building_ontology.ttl` | Building knowledge graph |
| `semantic/models/graph_encoder.pt` | Pre-trained GNN weights |
