#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-corl}"
if [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:/root/.mujoco/mujoco210/bin"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/corl_wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/corl_wandb_cache}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/root/autodl-tmp/d4rl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$D4RL_DATASET_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/offline/dtrd_dt/hopper/medium_v2.yaml}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UNROLLED_REWARD_UPDATE="${UNROLLED_REWARD_UPDATE:-true}"
DEVICE="${DEVICE:-cuda}"

for seed in $SEEDS; do
  echo "Running DTRD-DT on hopper-medium-v2 seed=${seed}"
  python -m algorithms.offline.dtrd_dt \
    --config_path "$CONFIG_PATH" \
    --train_seed "$seed" \
    --update_steps "$UPDATE_STEPS" \
    --eval_every "$EVAL_EVERY" \
    --eval_episodes "$EVAL_EPISODES" \
    --batch_size "$BATCH_SIZE" \
    --unrolled_reward_update "$UNROLLED_REWARD_UPDATE" \
    --device "$DEVICE"
done
