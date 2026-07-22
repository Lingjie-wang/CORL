#!/usr/bin/env bash
set -euo pipefail

# Faithful DTRD reproduction (bilevel meta-optimization), entry point
# algorithms.offline.atari_dtrd_checkpoints_ordered_wlj. Per-game hyper-parameters follow the
# official DTRD scripts (n_layer=2, per-game n_embd, per-game discrete flag,
# lr=1e-3, batch=64, drop_out=0.1, epochs=100).
#
# SMOKE=1 runs a fast correctness check (1 epoch, single game, single seed).

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export PYTHONPATH="${PYTHONPATH:-}"
export PYTHONPATH="$REPO_ROOT:$PYTHONPATH"

CONDA_ENV="${CONDA_ENV:-corlenv}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/outputs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$REPO_ROOT/outputs/wandb_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

SMOKE="${SMOKE:-0}"
if [ "$SMOKE" = "1" ]; then
  GAMES="${GAMES:-Breakout}"
  SEEDS="${SEEDS:-10}"
  EPOCHS="${EPOCHS:-1}"
else
  GAMES="${GAMES:-Breakout Seaquest Qbert Pong}"
  SEEDS="${SEEDS:-10 20 30}"
  EPOCHS="${EPOCHS:-100}"
fi

NUM_STEPS="${NUM_STEPS:-500000}"
NUM_BUFFERS="${NUM_BUFFERS:-50}"
TRAJECTORIES_PER_BUFFER="${TRAJECTORIES_PER_BUFFER:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-30}"
LEARNING_RATE="${LEARNING_RATE:-0.001}"
DROP_OUT="${DROP_OUT:-0.1}"
VAL_FRACTION="${VAL_FRACTION:-0.1}"
REDISTRIBUTE_LEARNING_RATE="${REDISTRIBUTE_LEARNING_RATE:-0.001}"
REDISTRIBUTE_STEP_SIZE="${REDISTRIBUTE_STEP_SIZE:-1000}"
REDISTRIBUTE_GAMMA="${REDISTRIBUTE_GAMMA:-0.9}"
TRAJECTORY_LAMB="${TRAJECTORY_LAMB:-0.01}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-}"
ATARI_DATA_DIR="${ATARI_DATA_DIR:-$REPO_ROOT/outputs/atari/dqn_replay}"
DATA_SOURCE="${DATA_SOURCE:-tfds}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds_checkpoints_ordered}"
TFDS_RUN="${TFDS_RUN:-1}"
TFDS_CHECKPOINT_SPLITS="${TFDS_CHECKPOINT_SPLITS:-all}"
TFDS_RAW_INPUT_PREFIX="${TFDS_RAW_INPUT_PREFIX:-$REPO_ROOT/outputs/atari/rl_unplugged_raw/atari_episodes_ordered}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"

tfds_dataset_exists() {
  local game="$1"
  local version_dir="$TFDS_DATA_DIR/rlu_atari_checkpoints_ordered/${game}_run_${TFDS_RUN}/1.1.0"
  [ -f "$version_dir/dataset_info.json" ] && find "$version_dir" -name 'rlu_atari_checkpoints_ordered-checkpoint_*.tfrecord-*' -type f -print -quit | grep -q .
}

# Per-game policy embedding size and redistribution type (official DTRD scripts).
n_embd_for_game() {
  case "$1" in
    Breakout) echo 32 ;;
    Seaquest) echo 128 ;;
    Qbert|Pong) echo 64 ;;
    *) echo 128 ;;
  esac
}
discrete_for_game() {
  case "$1" in
    Breakout|Seaquest) echo 1 ;;
    *) echo 0 ;;
  esac
}

if [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "dqn_replay" ]; then
  GAMES="$GAMES" ATARI_DATA_DIR="$ATARI_DATA_DIR" scripts/download_atari_dqn_replay_wlj.sh
elif [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "tfds" ]; then
  missing_games=""
  for game in $GAMES; do
    if tfds_dataset_exists "$game"; then
      echo "TFDS dataset already exists for $game"
    else
      missing_games="$missing_games $game"
    fi
  done
  if [ -n "$missing_games" ]; then
    GAMES="$missing_games" TFDS_DATA_DIR="$TFDS_DATA_DIR" TFDS_RUN="$TFDS_RUN" RAW_INPUT_PREFIX="$TFDS_RAW_INPUT_PREFIX" scripts/download_atari_tfds_checkpoints_ordered_wlj.sh
  fi
fi

for game in $GAMES; do
  if [ "$DATA_SOURCE" = "dqn_replay" ] && [ ! -d "$ATARI_DATA_DIR/$game/1/replay_logs" ]; then
    echo "Missing Atari dataset for $game: $ATARI_DATA_DIR/$game/1/replay_logs" >&2
    exit 1
  elif [ "$DATA_SOURCE" = "tfds" ] && [ "$DOWNLOAD_DATA" != "1" ] && ! tfds_dataset_exists "$game"; then
    echo "Missing TFDS Atari dataset for $game: $TFDS_DATA_DIR/rlu_atari_checkpoints_ordered/${game}_run_${TFDS_RUN}/1.1.0" >&2
    exit 1
  fi

  n_embd="$(n_embd_for_game "$game")"
  discrete="$(discrete_for_game "$game")"
  for seed in $SEEDS; do
    echo "Running Atari DTRD (faithful) on ${game} seed=${seed} n_embd=${n_embd} discrete=${discrete}"
    args=(
      --seed "$seed"
      --context_length "$CONTEXT_LENGTH"
      --epochs "$EPOCHS"
      --game "$game"
      --num_steps "$NUM_STEPS"
      --num_buffers "$NUM_BUFFERS"
      --trajectories_per_buffer "$TRAJECTORIES_PER_BUFFER"
      --batch_size "$BATCH_SIZE"
      --val_fraction "$VAL_FRACTION"
      --data_dir_prefix "$ATARI_DATA_DIR"
      --data_source "$DATA_SOURCE"
      --tfds_data_dir "$TFDS_DATA_DIR"
      --tfds_run "$TFDS_RUN"
      --tfds_checkpoint_splits "$TFDS_CHECKPOINT_SPLITS"
      --tfds_raw_input_prefix "$TFDS_RAW_INPUT_PREFIX"
      --learning_rate "$LEARNING_RATE"
      --drop_out "$DROP_OUT"
      --num_workers "$NUM_WORKERS"
      --device "$DEVICE"
      --n_layer 2
      --n_head 8
      --n_embd "$n_embd"
      --discrete_redistribute "$discrete"
      --redistribute_learning_rate "$REDISTRIBUTE_LEARNING_RATE"
      --redistribute_step_size "$REDISTRIBUTE_STEP_SIZE"
      --redistribute_gamma "$REDISTRIBUTE_GAMMA"
      --trajectory_lamb "$TRAJECTORY_LAMB"
      --eval_episodes "$EVAL_EPISODES"
    )
    if [ -n "$CHECKPOINTS_PATH" ]; then
      args+=(--checkpoints_path "$CHECKPOINTS_PATH")
    fi
    if [ "$DATA_SOURCE" = "tfds" ] && { [ "$DOWNLOAD_DATA" != "1" ] || tfds_dataset_exists "$game"; }; then
      args+=(--no-tfds_download)
    fi
    python -m algorithms.offline.atari_dtrd_checkpoints_ordered_wlj "${args[@]}"
  done
done
