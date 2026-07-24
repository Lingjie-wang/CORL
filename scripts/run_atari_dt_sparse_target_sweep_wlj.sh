#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

TARGETS="${TARGETS:-50 145 290}"

for target in $TARGETS; do
  echo "Running Seaquest sparse DT target sweep: eval_target_return=${target}"
  GAMES="${GAMES:-Seaquest}" \
  MODES="${MODES:-sparse}" \
  SEEDS="${SEEDS:-10 20 30}" \
  MODEL_TYPE="${MODEL_TYPE:-reward_conditioned}" \
  DATA_SOURCE="${DATA_SOURCE:-tfds}" \
  DOWNLOAD_DATA="${DOWNLOAD_DATA:-0}" \
  TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds_checkpoints_ordered}" \
  TFDS_CHECKPOINT_SPLITS="${TFDS_CHECKPOINT_SPLITS:-all}" \
  TFDS_SAMPLING_MODE="${TFDS_SAMPLING_MODE:-balanced}" \
  EVAL_TARGET_RETURN="$target" \
  EVAL_RTG_UPDATE="${EVAL_RTG_UPDATE:-delayed}" \
  WANDB_SUFFIX="target${target}" \
  ./scripts/run_atari_dt_checkpoints_ordered_wlj.sh
done
