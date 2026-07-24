#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export CONDA_ENV="${CONDA_ENV:-corl}"
export GAMES="${GAMES:-Seaquest}"
export MODES="${MODES:-dense sparse}"
export SEEDS="${SEEDS:-123 231 312}"
export DATA_SOURCE="${DATA_SOURCE:-tfds}"
export DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds_checkpoints_ordered}"
export TFDS_CHECKPOINT_SPLITS="${TFDS_CHECKPOINT_SPLITS:-all}"
export TFDS_SAMPLING_MODE="${TFDS_SAMPLING_MODE:-dt_replay}"
export NUM_STEPS="${NUM_STEPS:-500000}"
export NUM_BUFFERS="${NUM_BUFFERS:-50}"
export TRAJECTORIES_PER_BUFFER="${TRAJECTORIES_PER_BUFFER:-10}"
export EPOCHS="${EPOCHS:-5}"
export EVAL_EPISODES="${EVAL_EPISODES:-100}"
export EVAL_TARGET_RETURN="${EVAL_TARGET_RETURN:-290}"
export EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-}"

exec ./scripts/run_atari_dt_checkpoints_ordered_wlj.sh
