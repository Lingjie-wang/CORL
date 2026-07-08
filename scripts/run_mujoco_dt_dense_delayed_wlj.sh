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

ENVS="${ENVS:-halfcheetah-medium-v2 hopper-medium-v2 walker2d-medium-v2}"
MODES="${MODES:-dense delayed}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-5000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DEVICE="${DEVICE:-cuda}"

config_path_for_env() {
  case "$1" in
    halfcheetah-*-v2)
      echo "configs/offline/dt/halfcheetah/${1#halfcheetah-}" | sed 's/-v2$/_v2.yaml/; s/-/_/g'
      ;;
    hopper-*-v2)
      echo "configs/offline/dt/hopper/${1#hopper-}" | sed 's/-v2$/_v2.yaml/; s/-/_/g'
      ;;
    walker2d-*-v2)
      echo "configs/offline/dt/walker2d/${1#walker2d-}" | sed 's/-v2$/_v2.yaml/; s/-/_/g'
      ;;
    *)
      echo "Unsupported MuJoCo env: $1" >&2
      return 1
      ;;
  esac
}

label_for_mode() {
  case "$1" in
    dense)
      echo "original"
      ;;
    delayed)
      echo "delayed"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

for env_name in $ENVS; do
  config_path="$(config_path_for_env "$env_name")"
  if [ ! -f "$config_path" ]; then
    echo "Missing config for $env_name: $config_path" >&2
    exit 1
  fi

  for mode in $MODES; do
    label="$(label_for_mode "$mode")"
    for seed in $SEEDS; do
      echo "Running DT ${label} on ${env_name} seed=${seed}"
      python algorithms/offline/dt_wlj.py \
        --config_path "$config_path" \
        --env_name "$env_name" \
        --reward_mode "$mode" \
        --group "dt-${env_name}-${label}-multiseed-v0" \
        --name "DT-${label}" \
        --train_seed "$seed" \
        --update_steps "$UPDATE_STEPS" \
        --eval_every "$EVAL_EVERY" \
        --eval_episodes "$EVAL_EPISODES" \
        --device "$DEVICE"
    done
  done
done
