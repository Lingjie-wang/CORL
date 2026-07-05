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

METHODS="${METHODS:-markov gru}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DEVICE="${DEVICE:-cuda}"

config_path_for_method() {
  case "$1" in
    markov)
      echo "configs/offline/dtrd_dt/antmaze/medium_diverse_v2_markov.yaml"
      ;;
    gru)
      echo "configs/offline/dtrd_dt/antmaze/medium_diverse_v2_gru.yaml"
      ;;
    *)
      echo "Unknown method: $1" >&2
      return 1
      ;;
  esac
}

for method in $METHODS; do
  config_path="$(config_path_for_method "$method")"
  for seed in $SEEDS; do
    echo "Running ${method} DTRD on antmaze-medium-diverse-v2 seed=${seed}"
    python -m algorithms.offline.dtrd_dt \
      --config_path "$config_path" \
      --train_seed "$seed" \
      --update_steps "$UPDATE_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --eval_episodes "$EVAL_EPISODES" \
      --device "$DEVICE"
  done
done
