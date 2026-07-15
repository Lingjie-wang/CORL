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

export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="${WANDB_DIR:-$REPO_ROOT/outputs/wandb}"
export WANDB_CACHE_DIR="${WANDB_CACHE_DIR:-$REPO_ROOT/outputs/wandb_cache}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
mkdir -p "$WANDB_DIR" "$WANDB_CACHE_DIR"

GAMES="${GAMES:-Breakout Seaquest Qbert Pong}"
MODES="${MODES:-dense delayed}"
# Seeds aligned with run_atari_dtrd_wlj.sh so DT-baseline and DTRD runs pair up.
SEEDS="${SEEDS:-123 231 312}"
EPOCHS="${EPOCHS:-5}"
NUM_STEPS="${NUM_STEPS:-500000}"
NUM_BUFFERS="${NUM_BUFFERS:-50}"
TRAJECTORIES_PER_BUFFER="${TRAJECTORIES_PER_BUFFER:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-}"
ATARI_DATA_DIR="${ATARI_DATA_DIR:-$REPO_ROOT/outputs/atari/dqn_replay}"
DATA_SOURCE="${DATA_SOURCE:-dqn_replay}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/outputs/atari/tfds}"
TFDS_RUN="${TFDS_RUN:-1}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-1}"

tfds_dataset_exists() {
  local game="$1"
  local version_dir="$TFDS_DATA_DIR/rlu_atari/${game}_run_${TFDS_RUN}/1.3.0"
  [ -f "$version_dir/dataset_info.json" ] && find "$version_dir" -name 'rlu_atari-train.tfrecord-*' -type f -print -quit | grep -q .
}

if [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "dqn_replay" ]; then
  GAMES="$GAMES" ATARI_DATA_DIR="$ATARI_DATA_DIR" scripts/download_atari_dqn_replay_wlj.sh
elif [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "tfds" ]; then
  missing_games=""
  for game in $GAMES; do
    if tfds_dataset_exists "$game"; then
      echo "TFDS dataset already exists for $game: $TFDS_DATA_DIR/rlu_atari/${game}_run_${TFDS_RUN}/1.3.0"
    else
      missing_games="$missing_games $game"
    fi
  done
  if [ -n "$missing_games" ]; then
    GAMES="$missing_games" TFDS_DATA_DIR="$TFDS_DATA_DIR" TFDS_RUN="$TFDS_RUN" scripts/download_atari_tfds_wlj.sh
  fi
fi

context_length_for_game() {
  if [ -n "${CONTEXT_LENGTH:-}" ]; then
    echo "$CONTEXT_LENGTH"
  elif [ "$1" = "Pong" ]; then
    echo 50
  else
    echo 30
  fi
}

batch_size_for_game() {
  if [ -n "${BATCH_SIZE:-}" ]; then
    echo "$BATCH_SIZE"
  elif [ "$1" = "Pong" ]; then
    echo 512
  else
    echo 128
  fi
}

label_for_mode() {
  case "$1" in
    dense)
      echo "dense"
      ;;
    delayed)
      echo "delayed"
      ;;
    sparse)
      echo "sparse"
      ;;
    *)
      echo "$1"
      ;;
  esac
}

for game in $GAMES; do
  if [ "$DATA_SOURCE" = "dqn_replay" ] && [ ! -d "$ATARI_DATA_DIR/$game/1/replay_logs" ]; then
    echo "Missing Atari dataset for $game: $ATARI_DATA_DIR/$game/1/replay_logs" >&2
    echo "Run scripts/download_atari_dqn_replay_wlj.sh or set ATARI_DATA_DIR." >&2
    exit 1
  elif [ "$DATA_SOURCE" = "tfds" ] && [ "$DOWNLOAD_DATA" != "1" ] && ! tfds_dataset_exists "$game"; then
    echo "Missing TFDS Atari dataset for $game: $TFDS_DATA_DIR/rlu_atari/${game}_run_${TFDS_RUN}/1.3.0" >&2
    echo "Run scripts/download_atari_tfds_wlj.sh or set DOWNLOAD_DATA=1." >&2
    exit 1
  elif [ "$DATA_SOURCE" != "dqn_replay" ] && [ "$DATA_SOURCE" != "tfds" ]; then
    echo "Unsupported DATA_SOURCE=$DATA_SOURCE. Use dqn_replay or tfds." >&2
    exit 1
  fi

  context_length="$(context_length_for_game "$game")"
  batch_size="$(batch_size_for_game "$game")"
  for mode in $MODES; do
    label="$(label_for_mode "$mode")"
    for seed in $SEEDS; do
      echo "Running Atari DT ${label} on ${game} seed=${seed}"
      args=(
        --seed "$seed"
        --context_length "$context_length"
        --epochs "$EPOCHS"
        --model_type reward_conditioned
        --num_steps "$NUM_STEPS"
        --num_buffers "$NUM_BUFFERS"
        --game "$game"
        --batch_size "$batch_size"
        --trajectories_per_buffer "$TRAJECTORIES_PER_BUFFER"
        --data_dir_prefix "$ATARI_DATA_DIR"
        --data_source "$DATA_SOURCE"
        --reward_mode "$mode"
        --tfds_data_dir "$TFDS_DATA_DIR"
        --tfds_run "$TFDS_RUN"
        --num_workers "$NUM_WORKERS"
        --device "$DEVICE"
        --eval_episodes "$EVAL_EPISODES"
      )
      if [ -n "$CHECKPOINTS_PATH" ]; then
        args+=(--checkpoints_path "$CHECKPOINTS_PATH")
      fi
      if [ -n "$EVAL_EVERY_STEPS" ]; then
        args+=(--eval_every_steps "$EVAL_EVERY_STEPS")
      fi
      if [ -n "$EVAL_RTG_UPDATE" ]; then
        args+=(--eval_rtg_update "$EVAL_RTG_UPDATE")
      fi
      if [ "$DATA_SOURCE" = "tfds" ] && { [ "$DOWNLOAD_DATA" != "1" ] || tfds_dataset_exists "$game"; }; then
        args+=(--no-tfds_download)
      fi
      python -m algorithms.offline.atari_dt_wlj "${args[@]}"
    done
  done
done
