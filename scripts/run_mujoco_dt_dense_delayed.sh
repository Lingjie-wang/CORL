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
export WANDB_MODE="${WANDB_MODE:-online}" #* 默认 online，运行结果会上传到 wandb
export WANDB_DIR="${WANDB_DIR:-/root/autodl-tmp/corl_wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-/root/autodl-tmp/corl_wandb_cache}"
export D4RL_DATASET_DIR="${D4RL_DATASET_DIR:-/root/autodl-tmp/d4rl}"
export D4RL_SUPPRESS_IMPORT_ERROR="${D4RL_SUPPRESS_IMPORT_ERROR:-1}"
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

for env_name in $ENVS; do
  config_path="$(config_path_for_env "$env_name")"
  if [ ! -f "$config_path" ]; then
    echo "Missing config for $env_name: $config_path" >&2
    exit 1
  fi

  for mode in $MODES; do
    for seed in $SEEDS; do
      group="dt-${env_name}-${mode}-multiseed-v0"
      name="DT-${mode}"
      echo "Running env=${env_name} reward_mode=${mode} seed=${seed}"
      python algorithms/offline/dt.py \
        --config_path "$config_path" \
        --env_name "$env_name" \
        --reward_mode "$mode" \
        --group "$group" \
        --name "$name" \
        --train_seed "$seed" \
        --update_steps "$UPDATE_STEPS" \
        --eval_every "$EVAL_EVERY" \
        --eval_episodes "$EVAL_EPISODES" \
        --device "$DEVICE"
    done
  done
done
