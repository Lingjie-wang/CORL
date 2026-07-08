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

ENVS="${ENVS:-hopper-medium-v2 halfcheetah-medium-v2 walker2d-medium-v2}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
BATCH_SIZE="${BATCH_SIZE:-2048}"
UNROLLED_REWARD_UPDATE="${UNROLLED_REWARD_UPDATE:-true}"
DEVICE="${DEVICE:-cuda}"

config_path_for_env() {
  case "$1" in
    halfcheetah-medium-v2)
      echo "configs/offline/dtrd_dt/halfcheetah/medium_v2.yaml"
      ;;
    hopper-medium-v2)
      echo "configs/offline/dtrd_dt/hopper/medium_v2.yaml"
      ;;
    walker2d-medium-v2)
      echo "configs/offline/dtrd_dt/walker2d/medium_v2.yaml"
      ;;
    *)
      echo "Unsupported DTRD-DT MuJoCo env: $1" >&2
      return 1
      ;;
  esac
}

for env_name in $ENVS; do
  config_path="$(config_path_for_env "$env_name")"
  if [ ! -f "$config_path" ]; then
    echo "Missing config for $env_name: $config_path" >&2
    exit 1
  fi

  for seed in $SEEDS; do
    echo "Running DTRD-DT on ${env_name} seed=${seed}"
    python -m algorithms.offline.dtrd_dt_wlj \
      --config_path "$config_path" \
      --train_seed "$seed" \
      --update_steps "$UPDATE_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --eval_episodes "$EVAL_EPISODES" \
      --batch_size "$BATCH_SIZE" \
      --unrolled_reward_update "$UNROLLED_REWARD_UPDATE" \
      --device "$DEVICE"
  done
done
