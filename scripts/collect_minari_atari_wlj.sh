#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

CONDA_ENV="${CONDA_ENV:-corl}"
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV"
elif [ -f /root/miniconda3/etc/profile.d/conda.sh ]; then
  source /root/miniconda3/etc/profile.d/conda.sh
  conda activate "$CONDA_ENV"
fi

GAME="${GAME:-Seaquest}"
POLICY="${POLICY:-random}"
TOTAL_STEPS="${TOTAL_STEPS:-500000}"
SEED="${SEED:-0}"
MINARI_DATA_DIR="${MINARI_DATA_DIR:-$REPO_ROOT/data/minari}"
DATASET_ID="${DATASET_ID:-corl/$(printf '%s' "$GAME" | tr '[:upper:]' '[:lower:]')-${POLICY}-v0}"
CHECKPOINT_EVERY_EPISODES="${CHECKPOINT_EVERY_EPISODES:-50}"
REPEAT_ACTION_PROBABILITY="${REPEAT_ACTION_PROBABILITY:-0.0}"
FRAME_SKIP="${FRAME_SKIP:-4}"
NOOP_MAX="${NOOP_MAX:-30}"
AUTHOR="${AUTHOR:-yewei}"
AUTHOR_EMAIL="${AUTHOR_EMAIL:-}"
OVERWRITE="${OVERWRITE:-0}"

args=(
  --game "$GAME"
  --dataset_id "$DATASET_ID"
  --minari_data_dir "$MINARI_DATA_DIR"
  --total_steps "$TOTAL_STEPS"
  --seed "$SEED"
  --policy "$POLICY"
  --checkpoint_every_episodes "$CHECKPOINT_EVERY_EPISODES"
  --repeat_action_probability "$REPEAT_ACTION_PROBABILITY"
  --frame_skip "$FRAME_SKIP"
  --noop_max "$NOOP_MAX"
  --author "$AUTHOR"
)

if [ -n "$AUTHOR_EMAIL" ]; then
  args+=(--author_email "$AUTHOR_EMAIL")
fi
if [ "$OVERWRITE" = "1" ]; then
  args+=(--overwrite)
fi

python scripts/collect_minari_atari_wlj.py "${args[@]}"
