#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Load repo-local wandb credentials if present. This keeps your personal
# WANDB_API_KEY scoped to this directory only: it overrides the global
# ~/.netrc for runs launched here, without touching that shared file.
if [ -f "$REPO_ROOT/.wandb.env" ]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.wandb.env"
  set +a
fi

CONDA_ENV="${CONDA_ENV:-corl}"
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

GAMES="${GAMES:-Seaquest}"
MODES="${MODES:-dense}"
# Seeds aligned with run_atari_dtrd_wlj.sh so DT-baseline and DTRD runs pair up.
SEEDS="${SEEDS:-10 20 30}"
EPOCHS="${EPOCHS:-5}"
MODEL_TYPE="${MODEL_TYPE:-reward_conditioned}"
NUM_STEPS="${NUM_STEPS:-500000}"
NUM_BUFFERS="${NUM_BUFFERS:-50}"
TRAJECTORIES_PER_BUFFER="${TRAJECTORIES_PER_BUFFER:-10}"
DEVICE="${DEVICE:-cuda}"
NUM_WORKERS="${NUM_WORKERS:-4}"
EVAL_EPISODES="${EVAL_EPISODES:-10}"
EVAL_TARGET_RETURN="${EVAL_TARGET_RETURN:-290}"
EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-}"
EVAL_EVERY_STEPS="${EVAL_EVERY_STEPS:-}"
CHECKPOINTS_PATH="${CHECKPOINTS_PATH:-}"
WANDB_SUFFIX="${WANDB_SUFFIX:-}"
ATARI_DATA_DIR="${ATARI_DATA_DIR:-$REPO_ROOT/outputs/atari/dqn_replay}"
DATA_SOURCE="${DATA_SOURCE:-tfds}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds_checkpoints_ordered}"
TFDS_RUN="${TFDS_RUN:-1}"
TFDS_CHECKPOINT_SPLITS="${TFDS_CHECKPOINT_SPLITS:-all}"
TFDS_SAMPLING_MODE="${TFDS_SAMPLING_MODE:-balanced}"
TFDS_SAMPLING_SEED="${TFDS_SAMPLING_SEED:-}"
TFDS_RAW_INPUT_PREFIX="${TFDS_RAW_INPUT_PREFIX:-$REPO_ROOT/outputs/atari/rl_unplugged_raw/atari_episodes_ordered}"
MINARI_DATA_DIR="${MINARI_DATA_DIR:-$REPO_ROOT/data/minari}"
MINARI_DATASET_ID="${MINARI_DATASET_ID:-}"
MINARI_DOWNLOAD="${MINARI_DOWNLOAD:-0}"
MINARI_NUM_SHARDS="${MINARI_NUM_SHARDS:-50}"
MINARI_DATASET_PREFIX="${MINARI_DATASET_PREFIX:-}"
MINARI_SAMPLING_MODE="${MINARI_SAMPLING_MODE:-balanced}"
MINARI_SAMPLING_SEED="${MINARI_SAMPLING_SEED:-}"
HDF5_DATA_DIR="${HDF5_DATA_DIR:-$REPO_ROOT/data/atari/dqn_replay_hdf5}"
HDF5_SHARD_PATHS="${HDF5_SHARD_PATHS:-}"
HDF5_NUM_SHARDS="${HDF5_NUM_SHARDS:-50}"
HDF5_SAMPLING_MODE="${HDF5_SAMPLING_MODE:-balanced}"
HDF5_SAMPLING_SEED="${HDF5_SAMPLING_SEED:-}"
DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"

tfds_dataset_exists() {
  local game="$1"
  local version_dir="$TFDS_DATA_DIR/rlu_atari_checkpoints_ordered/${game}_run_${TFDS_RUN}/1.1.0"
  [ -f "$version_dir/dataset_info.json" ] && find "$version_dir" -name 'rlu_atari_checkpoints_ordered-checkpoint_*.tfrecord-*' -type f -print -quit | grep -q .
}

atari_rom_exists() {
  local game="$1"
  python - "$game" <<'PY'
import sys
import atari_py

game = sys.argv[1].lower()
try:
    atari_py.get_game_path(game)
except Exception:
    sys.exit(1)
PY
}

if [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "dqn_replay" ]; then
  GAMES="$GAMES" ATARI_DATA_DIR="$ATARI_DATA_DIR" scripts/download_atari_dqn_replay_wlj.sh
elif [ "$DOWNLOAD_DATA" = "1" ] && [ "$DATA_SOURCE" = "tfds" ]; then
  missing_games=""
  for game in $GAMES; do
    if tfds_dataset_exists "$game"; then
      echo "TFDS dataset already exists for $game: $TFDS_DATA_DIR/rlu_atari_checkpoints_ordered/${game}_run_${TFDS_RUN}/1.1.0"
    else
      missing_games="$missing_games $game"
    fi
  done
  if [ -n "$missing_games" ]; then
    GAMES="$missing_games" TFDS_DATA_DIR="$TFDS_DATA_DIR" TFDS_RUN="$TFDS_RUN" RAW_INPUT_PREFIX="$TFDS_RAW_INPUT_PREFIX" scripts/download_atari_tfds_checkpoints_ordered_wlj.sh
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

eval_rtg_update_for_mode() {
  if [ -n "$EVAL_RTG_UPDATE" ]; then
    echo "$EVAL_RTG_UPDATE"
  elif [ "$1" = "sparse" ] || [ "$1" = "delayed" ]; then
    echo "delayed"
  else
    echo "clipped_dense"
  fi
}

for game in $GAMES; do
  if [ "$DATA_SOURCE" = "dqn_replay" ] && [ ! -d "$ATARI_DATA_DIR/$game/1/replay_logs" ]; then
    echo "Missing Atari dataset for $game: $ATARI_DATA_DIR/$game/1/replay_logs" >&2
    echo "Run scripts/download_atari_dqn_replay_wlj.sh or set ATARI_DATA_DIR." >&2
    exit 1
  elif [ "$DATA_SOURCE" = "tfds" ] && [ "$DOWNLOAD_DATA" != "1" ] && ! tfds_dataset_exists "$game"; then
    echo "Missing TFDS Atari dataset for $game: $TFDS_DATA_DIR/rlu_atari_checkpoints_ordered/${game}_run_${TFDS_RUN}/1.1.0" >&2
    echo "Run scripts/download_atari_tfds_checkpoints_ordered_wlj.sh or set DOWNLOAD_DATA=1." >&2
    exit 1
  elif [ "$DATA_SOURCE" = "hdf5" ] && [ -z "$HDF5_SHARD_PATHS" ]; then
    for shard_idx in $(seq 1 "$HDF5_NUM_SHARDS"); do
      shard_name="$(printf '%02d' "$shard_idx")"
      if [ ! -f "$HDF5_DATA_DIR/$game/epoch_${shard_name}.hdf5" ]; then
        echo "Missing raw HDF5 Atari shard: $HDF5_DATA_DIR/$game/epoch_${shard_name}.hdf5" >&2
        echo "Run scripts/collect_atari_dqn_replay_hdf5_wlj.sh first." >&2
        exit 1
      fi
    done
  elif [ "$DATA_SOURCE" != "dqn_replay" ] && [ "$DATA_SOURCE" != "tfds" ] && [ "$DATA_SOURCE" != "minari" ] && [ "$DATA_SOURCE" != "hdf5" ]; then
    echo "Unsupported DATA_SOURCE=$DATA_SOURCE. Use dqn_replay, tfds, minari, or hdf5." >&2
    exit 1
  fi
  if [ "$EVAL_EPISODES" -gt 0 ] && ! atari_rom_exists "$game"; then
    echo "Missing Atari ROM for $game; online eval would fail after training." >&2
    echo "Install ROMs first with: CONDA_ENV=$CONDA_ENV scripts/setup_atari_roms_wlj.sh" >&2
    echo "Or skip online eval with: EVAL_EPISODES=0 ./scripts/run_atari_dt_checkpoints_ordered_wlj.sh" >&2
    exit 1
  fi

  context_length="$(context_length_for_game "$game")"
  batch_size="$(batch_size_for_game "$game")"
  for mode in $MODES; do
    label="$(label_for_mode "$mode")"
    eval_rtg_update="$(eval_rtg_update_for_mode "$mode")"
    for seed in $SEEDS; do
      echo "Running Atari DT ${label} on ${game} seed=${seed}"
      args=(
        --seed "$seed"
        --context_length "$context_length"
        --epochs "$EPOCHS"
        --model_type "$MODEL_TYPE"
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
        --tfds_checkpoint_splits "$TFDS_CHECKPOINT_SPLITS"
        --tfds_sampling_mode "$TFDS_SAMPLING_MODE"
        --tfds_raw_input_prefix "$TFDS_RAW_INPUT_PREFIX"
        --minari_data_dir "$MINARI_DATA_DIR"
        --minari_num_shards "$MINARI_NUM_SHARDS"
        --minari_sampling_mode "$MINARI_SAMPLING_MODE"
        --hdf5_data_dir "$HDF5_DATA_DIR"
        --hdf5_num_shards "$HDF5_NUM_SHARDS"
        --hdf5_sampling_mode "$HDF5_SAMPLING_MODE"
        --num_workers "$NUM_WORKERS"
        --device "$DEVICE"
        --eval_episodes "$EVAL_EPISODES"
      )
      if [ -n "$MINARI_DATASET_ID" ]; then
        args+=(--minari_dataset_id "$MINARI_DATASET_ID")
      fi
      if [ -n "$MINARI_DATASET_PREFIX" ]; then
        args+=(--minari_dataset_prefix "$MINARI_DATASET_PREFIX")
      fi
      if [ -n "$MINARI_SAMPLING_SEED" ]; then
        args+=(--minari_sampling_seed "$MINARI_SAMPLING_SEED")
      fi
      if [ -n "$HDF5_SHARD_PATHS" ]; then
        args+=(--hdf5_shard_paths "$HDF5_SHARD_PATHS")
      fi
      if [ -n "$HDF5_SAMPLING_SEED" ]; then
        args+=(--hdf5_sampling_seed "$HDF5_SAMPLING_SEED")
      fi
      if [ "$MINARI_DOWNLOAD" = "1" ]; then
        args+=(--minari_download)
      else
        args+=(--no-minari_download)
      fi
      if [ -n "$CHECKPOINTS_PATH" ]; then
        args+=(--checkpoints_path "$CHECKPOINTS_PATH")
      fi
      if [ -n "$WANDB_SUFFIX" ]; then
        args+=(--wandb_suffix "$WANDB_SUFFIX")
      fi
      if [ -n "$EVAL_EVERY_STEPS" ]; then
        args+=(--eval_every_steps "$EVAL_EVERY_STEPS")
      fi
      if [ -n "$EVAL_TARGET_RETURN" ]; then
        args+=(--eval_target_return "$EVAL_TARGET_RETURN")
      fi
      if [ -n "$TFDS_SAMPLING_SEED" ]; then
        args+=(--tfds_sampling_seed "$TFDS_SAMPLING_SEED")
      fi
      if [ -n "$eval_rtg_update" ]; then
        args+=(--eval_rtg_update "$eval_rtg_update")
      fi
      if [ "$DATA_SOURCE" = "tfds" ] && { [ "$DOWNLOAD_DATA" != "1" ] || tfds_dataset_exists "$game"; }; then
        args+=(--no-tfds_download)
      fi
      python -m algorithms.offline.atari_dt_checkpoints_ordered_wlj "${args[@]}"
    done
  done
done
