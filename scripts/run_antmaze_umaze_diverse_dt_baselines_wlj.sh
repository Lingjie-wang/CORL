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
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR" "$D4RL_DATASET_DIR"

CONFIG_PATH="${CONFIG_PATH:-configs/offline/dt/antmaze/umaze_diverse_v2.yaml}"
# MODES="${MODES:-dense delayed}" # dense 会在 W&B 中命名为 DT-original；delayed 会在 W&B 中命名为 DT-delayed
MODES="${MODES:-dense}"
SEEDS="${SEEDS:-10}"
UPDATE_STEPS="${UPDATE_STEPS:-100000}"
EVAL_EVERY="${EVAL_EVERY:-10000}"
EVAL_EPISODES="${EVAL_EPISODES:-100}"
DEVICE="${DEVICE:-cuda}"

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

for mode in $MODES; do
  label="$(label_for_mode "$mode")"
  for seed in $SEEDS; do
    echo "Running DT ${label} on antmaze-umaze-diverse-v2 seed=${seed}"
    python algorithms/offline/dt_wlj.py \
      --config_path "$CONFIG_PATH" \
      --reward_mode "$mode" \
      --group "dt-antmaze-umaze-diverse-v2-${label}-multiseed-v0" \
      --name "DT-${label}" \
      --train_seed "$seed" \
      --update_steps "$UPDATE_STEPS" \
      --eval_every "$EVAL_EVERY" \
      --eval_episodes "$EVAL_EPISODES" \
      --device "$DEVICE"
  done
done
