#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GAME="${GAME:-Seaquest}"

GAMES="$GAME" \
MODES="${MODES:-dense}" \
SEEDS="${SEEDS:-10 20 30}" \
DATA_SOURCE=hdf5 \
HDF5_DATA_DIR="${HDF5_DATA_DIR:-$REPO_ROOT/data/atari/dqn_replay_hdf5}" \
HDF5_NUM_SHARDS="${HDF5_NUM_SHARDS:-50}" \
HDF5_SAMPLING_MODE="${HDF5_SAMPLING_MODE:-balanced}" \
NUM_STEPS="${NUM_STEPS:-500000}" \
EVAL_TARGET_RETURN="${EVAL_TARGET_RETURN:-1450}" \
EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-dense}" \
WANDB_SUFFIX="${WANDB_SUFFIX:-hdf5-dqn-raw}" \
./scripts/run_atari_dt_checkpoints_ordered_wlj.sh
