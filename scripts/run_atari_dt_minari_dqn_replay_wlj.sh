#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

echo "Deprecated: Minari DQN replay shards store image observations as JPEG bytes." >&2
echo "Use lossless raw uint8 HDF5 shards instead:" >&2
echo "  ./scripts/run_atari_dt_hdf5_dqn_replay_wlj.sh" >&2
exit 1

GAME="${GAME:-Seaquest}"
GAME_LOWER="$(printf '%s' "$GAME" | tr '[:upper:]' '[:lower:]')"

GAMES="$GAME" \
MODES="${MODES:-dense}" \
SEEDS="${SEEDS:-10 20 30}" \
DATA_SOURCE=minari \
MINARI_DATA_DIR="${MINARI_DATA_DIR:-$REPO_ROOT/data/minari}" \
MINARI_DATASET_PREFIX="${MINARI_DATASET_PREFIX:-corl/${GAME_LOWER}-dqn-epoch}" \
MINARI_NUM_SHARDS="${MINARI_NUM_SHARDS:-50}" \
MINARI_SAMPLING_MODE="${MINARI_SAMPLING_MODE:-balanced}" \
NUM_STEPS="${NUM_STEPS:-500000}" \
EVAL_TARGET_RETURN="${EVAL_TARGET_RETURN:-1450}" \
EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-dense}" \
WANDB_SUFFIX="${WANDB_SUFFIX:-minari-dqn-raw}" \
./scripts/run_atari_dt_checkpoints_ordered_wlj.sh
