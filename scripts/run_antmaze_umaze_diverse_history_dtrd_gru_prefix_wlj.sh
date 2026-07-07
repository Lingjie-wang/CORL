#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-corlenv}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

MUJOCO_BIN="${MUJOCO_BIN:-$HOME/.mujoco/mujoco210/bin}"
if [ -d "$MUJOCO_BIN" ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:$MUJOCO_BIN"
fi
if [ -d /usr/lib/nvidia ]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/usr/lib/nvidia"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/outputs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$REPO_ROOT/outputs/wandb_cache}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-$REPO_ROOT/outputs/d4rl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$D4RL_DATASET_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/offline/dtrd_dt/antmaze/umaze_diverse_v2_gru_prefix.yaml}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DEVICE="${DEVICE:-cuda}"

for seed in $SEEDS; do
  echo "Running History-DTRD-GRU-Prefix on antmaze-umaze-diverse-v2 seed=${seed}"
  python -m algorithms.offline.dtrd_dt_prefix_wlj \
    --config_path "$CONFIG_PATH" \
    --train_seed "$seed" \
    --update_steps "$UPDATE_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --eval_episodes "$EVAL_EPISODES" \
    --device "$DEVICE"
done
