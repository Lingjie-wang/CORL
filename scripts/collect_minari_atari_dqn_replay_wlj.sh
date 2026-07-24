#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Deprecated: Minari stores image observations as JPEG bytes in this setup." >&2
echo "Use lossless raw uint8 HDF5 collection instead:" >&2
echo "  ./scripts/collect_atari_dqn_replay_hdf5_wlj.sh" >&2
exit 1

CONDA_ENV="${CONDA_ENV:-corl}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

GAME="${GAME:-Seaquest}"
SEED="${SEED:-0}"
DEVICE="${DEVICE:-cuda}"
MINARI_DATA_DIR="${MINARI_DATA_DIR:-$REPO_ROOT/data/minari}"
DATASET_PREFIX="${DATASET_PREFIX:-corl/$(printf '%s' "$GAME" | tr '[:upper:]' '[:lower:]')-dqn-epoch}"
NUM_EPOCHS="${NUM_EPOCHS:-50}"
STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-1000000}"
REPLAY_CAPACITY="${REPLAY_CAPACITY:-1000000}"
LEARNING_STARTS="${LEARNING_STARTS:-50000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TRAIN_FREQUENCY="${TRAIN_FREQUENCY:-4}"
TARGET_UPDATE_INTERVAL="${TARGET_UPDATE_INTERVAL:-10000}"
GAMMA="${GAMMA:-0.99}"
LEARNING_RATE="${LEARNING_RATE:-2.5e-4}"
EPSILON_DECAY_STEPS="${EPSILON_DECAY_STEPS:-1000000}"
REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY:-0.0}"
FRAME_SKIP="${FRAME_SKIP:-4}"
NOOP_MAX="${NOOP_MAX:-30}"
MODEL_DIR="${MODEL_DIR:-$REPO_ROOT/outputs/atari/minari_dqn}"
LOG_EVERY_STEPS="${LOG_EVERY_STEPS:-10000}"
AUTHOR="${AUTHOR:-yewei}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-}"
OVERWRITE="${OVERWRITE:-0}"
CLIP_TRAINING_REWARD="${CLIP_TRAINING_REWARD:-1}"

args=(
  --game "$GAME"
  --seed "$SEED"
  --device "$DEVICE"
  --minari_data_dir "$MINARI_DATA_DIR"
  --dataset_prefix "$DATASET_PREFIX"
  --num_epochs "$NUM_EPOCHS"
  --steps_per_epoch "$STEPS_PER_EPOCH"
  --replay_capacity "$REPLAY_CAPACITY"
  --learning_starts "$LEARNING_STARTS"
  --batch_size "$BATCH_SIZE"
  --train_frequency "$TRAIN_FREQUENCY"
  --target_update_interval "$TARGET_UPDATE_INTERVAL"
  --gamma "$GAMMA"
  --learning_rate "$LEARNING_RATE"
  --epsilon_decay_steps "$EPSILON_DECAY_STEPS"
  --repeat_action_probability "$REPEAT_ACTION_PROBABILITY"
  --frame_skip "$FRAME_SKIP"
  --noop_max "$NOOP_MAX"
  --model_dir "$MODEL_DIR"
  --log_every_steps "$LOG_EVERY_STEPS"
  --author "$AUTHOR"
)

if [ -n "$AUTHOR_EMAIL" ]; then
  args+=(--author_email "$AUTHOR_EMAIL")
fi
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi
if [ "$CLIP_TRAINING_REWARD" = "1" ]; then
  args+=(--clip_training_reward)
else
  args+=(--no-clip_training_reward)
fi

python scripts/collect_minari_atari_dqn_replay_wlj.py "${args[@]}"
