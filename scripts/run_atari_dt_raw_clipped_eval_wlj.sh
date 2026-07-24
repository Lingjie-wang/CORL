#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

export GAMES="${GAMES:-Seaquest}"
export MODES="${MODES:-dense sparse}"
export SEEDS="${SEEDS:-10 20 30}"
export MODEL_TYPE="${MODEL_TYPE:-reward_conditioned}"
export DATA_SOURCE="${DATA_SOURCE:-tfds}"
export DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}"
export TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds_checkpoints_ordered}"
export TFDS_CHECKPOINT_SPLITS="${TFDS_CHECKPOINT_SPLITS:-all}"
export TFDS_SAMPLING_MODE="${TFDS_SAMPLING_MODE:-balanced}"
export EVAL_TARGET_RETURN="${EVAL_TARGET_RETURN:-290}"
export EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-}"
export WANDB_SUFFIX="${WANDB_SUFFIX:-rawclipped}"

exec ./scripts/run_atari_dt_checkpoints_ordered_wlj.sh
