# ParallelRL HVAC Reference Guide

This document provides a comprehensive overview of all available RL algorithms, building environments, and weather conditions in the ParallelRL project.

---

## 🤖 Available Algorithms (10)

### Deep RL Algorithms

| Algorithm | Type | Description | Script |
|-----------|------|-------------|--------|
| **DQN** | Single-Agent | Deep Q-Network with target network, replay buffer, ε-greedy | `algorithms/dqn/` |
| **DDQN** | Single-Agent | Double DQN (reduced overestimation) | `algorithms/ddqn/` |
| **PPO** | Single-Agent | Proximal Policy Optimization | `algorithms/ppo/` |
| **SAC** | Single-Agent | Soft Actor-Critic (continuous actions) | `algorithms/sac/` |
| **A3C** | Single-Agent | Asynchronous Advantage Actor-Critic | `algorithms/a3c/` |

### Multi-Agent DQN Variants

| Variant | AC Reward | Fan Reward | Use Case |
|---------|-----------|------------|----------|
| **MADQN Fully Cooperative** | Global (shared) | Global (shared) | Collaboration |
| **MADQN Fully Competitive** | AC-specific | Fan-specific | Individual optimization |
| **MADQN Part. Cooperative** | Global + AC bonus | Global + Fan bonus | Mixed |
| **MADQN Part. Competitive** | AC-specific + global | Fan-specific + global | Mixed |

### Baseline Controllers

| Controller | Description |
|------------|-------------|
| **On-Off** | Simple thermostat-like control |
| **Setpoint** | Fixed setpoint controller |
| **RBC** | Rule-Based Controller |
| **Adaptive RBC** | Adaptive rule-based control |

### 🧠 Semantic/GNN Module (Advanced)

The `semantic/` folder contains a **Knowledge Graph + Graph Neural Network** enhancement:

| Component | File | Description |
|-----------|------|-------------|
| **Building Ontology** | `ontology/building_ontology.ttl` | RDF/Turtle knowledge graph of building structure |
| **GNN Encoder** | `source/gnn_encoder.py` | 3-layer Graph Convolutional Network |
| **Graph Builder** | `source/graph_builder.py` | Converts ontology to PyTorch graph |
| **Env Wrapper** | `env_wrappers.py` | Wraps sinergym env with semantic state |
| **Pre-trained Model** | `models/graph_encoder.pt` | Trained GNN weights |

**How it works:**
1. Building structure is represented as a **knowledge graph** (rooms, sensors, actuators, properties)
2. Sensor readings are mapped to graph node features
3. **GNN encodes** the graph into a latent vector `z_t`
4. RL agent receives `[observation, z_t]` as enhanced state

```python
# Usage example
from semantic.env_wrappers import SemanticGraphWrappedEnv
from semantic.source.semantic_state_provider import SemanticStateProvider

provider = SemanticStateProvider(...)
wrapped_env = SemanticGraphWrappedEnv(base_env, provider, mode="concat")
```

## 🏢 Building Environments (8 Types)

| Building ID | Size | Description | Comfort Model |
|-------------|------|-------------|---------------|
| `A403small` | Small | Small office room | Temperature-based |
| `A403medium` | Medium | Medium office room | Temperature-based |
| `A403large` | Large | Large office room | Temperature-based |
| `A403smallfanger` | Small | Small office room | **PMV/Fanger** |
| `A403mediumfanger` | Medium | Medium office room | **PMV/Fanger** ⭐ |
| `A403largefanger` | Large | Large office room | **PMV/Fanger** |
| `A403mediumwindow` | Medium | With window fan control | Temperature-based |
| `A403New` / `A403v3` | Medium | Updated building model | Mixed |

> ⭐ **Recommended**: `A403mediumfanger` - Uses Fanger PMV for thermal comfort

---

## 🌡️ Weather/Climate Options (4)

| Season | Location | Typical Temp | Description |
|--------|----------|--------------|-------------|
| `hot` | Arizona, USA | 30-45°C | Hot desert climate |
| `cool` | Washington, USA | 5-20°C | Cool temperate climate |
| `mixed` | Chicago, USA | -5 to 35°C | Four-season climate |
| `ankara` | Ankara, Turkey | -10 to 35°C | Continental climate |

---

## 🎮 Action Space (40 Discrete Actions)

The agent controls **HVAC setpoint** (10 options) × **Fan speed** (4 options) = 40 actions

### HVAC Setpoints
| Actions 0-3 | Actions 4-7 | ... | Actions 32-35 | Actions 36-39 |
|-------------|-------------|-----|---------------|---------------|
| 21-22°C | 22-23°C | ... | 29-30°C | **OFF** |

### Fan Speeds
| Level | Speed |
|-------|-------|
| 0 | 0% (Off) |
| 1 | 50% (Low) |
| 2 | 75% (Medium) |
| 3 | 100% (High) |

---

## 📊 Observation Space (15 Variables)

| Variable | Description | Range |
|----------|-------------|-------|
| `month` | Month of year | 1-12 |
| `day_of_month` | Day | 1-31 |
| `hour` | Hour of day | 0-23 |
| `outdoor_temperature` | Outside temp (°C) | -25 to 50 |
| `outdoor_humidity` | Outside humidity (%) | 0-100 |
| `htg_setpoint` | Heating setpoint | 5-25 |
| `clg_setpoint` | Cooling setpoint | 20-50 |
| `air_temperature` | Indoor temp (°C) | 15-35 |
| `air_humidity` | Indoor humidity (%) | 20-80 |
| `people_occupant` | # of occupants | 0-10 |
| `air_co2` | CO2 level (ppm) | 400-2000 |
| `window_fan_energy` | Fan power (J) | 0-100k |
| `pmv` | Thermal comfort index | -3 to +3 |
| `ppd` | % dissatisfied | 5-100 |
| `total_electricity_HVAC` | HVAC power (J) | 0-2M |

---

## 🎯 Reward Components

```
R = -λ_E × energy - co2_weight × co2_penalty - pmv_weight × |PMV| - switching_penalty
```

| Weight | Default | Description |
|--------|---------|-------------|
| `co2_weight` | 0.5 | Importance of air quality |
| `pmv_weight` | 0.4 | Importance of thermal comfort |
| `lambda_energy` | 1/1.6M | Energy consumption penalty |
| `switching_penalty` | 0 | Penalize frequent action changes |

---

## 🚀 Quick Start Examples

### Training
```bash
# Train DQN on hot climate for 5 episodes
conda activate hvac-rl
export PYTHONPATH=~/EnergyPlus-24.1.0:$PYTHONPATH

python train_standalone.py \
  --episodes 5 \
  --season hot \
  --env-id A403mediumfanger \
  --co2-weight 0.5 \
  --pmv-weight 0.4
```

### Evaluation
```bash
# Evaluate trained model
python evaluate_standalone.py \
  --model-path ./standalone_results/.../dqn_best.pth \
  --season hot
```

### Cross-Climate Testing
```bash
# Train on hot, test on cool
python train_standalone.py --episodes 10 --season hot
python evaluate_standalone.py --model-path .../dqn_best.pth --season cool
python evaluate_standalone.py --model-path .../dqn_best.pth --season mixed
```

---

## 📁 File Locations

| Purpose | Path |
|---------|------|
| Training script | `RL-HVAC/train_standalone.py` |
| Evaluation script | `RL-HVAC/evaluate_standalone.py` |
| DQN implementation | `RL-HVAC/algorithms/dqn/dqn.py` |
| MADQN experiments | `RL-HVAC/experiments/madqn/` |
| Building configs | `sinergym-a403/sinergym/data/default_configuration/` |
| Reward functions | `RL-HVAC/environments/reward.py` |
