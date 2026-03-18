#!/usr/bin/env bash
set -euo pipefail

# Train semantic GNN graphs across multiple environments/seasons
# and save each run into a unique folder so checkpoints are not overwritten.
#
# Usage:
#   cd /storage/Master/ParallelRL/RL-HVAC
#   source ~/.bashrc
#   conda activate hvac-rl
#   bash semantic/train_gnn_multi_env.sh
#
# Optional overrides:
#   CHUNKS=6 EPOCHS=40 LR=1e-3 bash semantic/train_gnn_multi_env.sh

ROOT_DIR="/storage/Master/ParallelRL/RL-HVAC"
TRAIN_SCRIPT="${ROOT_DIR}/semantic/train_gnn.py"
MODELS_DIR="${ROOT_DIR}/semantic/models"
OUT_BASE="${MODELS_DIR}/graphs"

CHUNKS="${CHUNKS:-8}"
EPOCHS="${EPOCHS:-60}"
LR="${LR:-1e-3}"

ENV_IDS=(
#  "A403smallfanger"
  "A403mediumfanger"
#  "A403largefanger"
)
SEASONS=(
  "hot"
  "cool"
  "mixed"
)

mkdir -p "${OUT_BASE}"
mkdir -p "${MODELS_DIR}"

echo "Starting semantic graph batch training"
echo "chunks=${CHUNKS} epochs=${EPOCHS} lr=${LR}"
echo

for env_id in "${ENV_IDS[@]}"; do
  for season in "${SEASONS[@]}"; do
    run_name="${env_id}_${season}"
    out_dir="${OUT_BASE}/${run_name}"

    echo "==> Training ${run_name}"
    python "${TRAIN_SCRIPT}" \
      --env-id "${env_id}" \
      --season "${season}" \
      --chunks "${CHUNKS}" \
      --epochs "${EPOCHS}" \
      --lr "${LR}"

    mkdir -p "${out_dir}"
    cp "${MODELS_DIR}/graph_encoder.pt" "${out_dir}/graph_encoder.pt"
    cp "${MODELS_DIR}/target_stats.json" "${out_dir}/target_stats.json"
    echo "Saved: ${out_dir}"
    echo
  done
done

echo "Done. Stored graph checkpoints:"
find "${OUT_BASE}" -maxdepth 2 -type f | sort
