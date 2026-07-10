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

GAMES="${GAMES:-Breakout Seaquest Qbert Pong}"
TFDS_RUN="${TFDS_RUN:-1}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/outputs/atari/tfds}"

mkdir -p "$TFDS_DATA_DIR"

for game in $GAMES; do
  echo "Downloading/preparing TFDS rlu_atari/${game}_run_${TFDS_RUN} to $TFDS_DATA_DIR"
  python - <<PY
import tensorflow_datasets as tfds

builder = tfds.builder("rlu_atari/${game}_run_${TFDS_RUN}", data_dir="${TFDS_DATA_DIR}")
builder.download_and_prepare()
print(builder.info)
PY
done
