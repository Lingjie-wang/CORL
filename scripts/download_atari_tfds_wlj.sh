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

GAMES="${GAMES:-Seaquest}"
TFDS_RUN="${TFDS_RUN:-1}"
TFDS_DATA_DIR="${TFDS_DATA_DIR:-$REPO_ROOT/data/atari/tfds}"
RAW_DATA_DIR="${RAW_DATA_DIR:-$REPO_ROOT/outputs/atari/rl_unplugged_raw}"
RAW_INPUT_PREFIX="${RAW_INPUT_PREFIX:-$RAW_DATA_DIR/atari_episodes_ordered}"
DOWNLOAD_RAW_SHARDS="${DOWNLOAD_RAW_SHARDS:-1}"
USE_LOCAL_RAW_SHARDS="${USE_LOCAL_RAW_SHARDS:-1}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-4}"
GCS_ATARI_BASE_URL="${GCS_ATARI_BASE_URL:-https://storage.googleapis.com/rl_unplugged/atari_episodes_ordered}"
CURL_RETRIES="${CURL_RETRIES:-50}"
CURL_CONNECT_TIMEOUT="${CURL_CONNECT_TIMEOUT:-30}"
CURL_SPEED_LIMIT="${CURL_SPEED_LIMIT:-1024}"
CURL_SPEED_TIME="${CURL_SPEED_TIME:-300}"

mkdir -p "$TFDS_DATA_DIR" "$RAW_INPUT_PREFIX"

num_shards_for_game() {
  case "$1" in
    Carnival|Gravitar|StarGunner)
      echo 49
      ;;
    *)
      echo 50
      ;;
  esac
}

download_one_shard() {
  local game="$1"
  local run="$2"
  local num_shards="$3"
  local shard="$4"
  local filename="run_${run}-${shard}-of-$(printf '%05d' "$num_shards")"
  local game_dir="$RAW_INPUT_PREFIX/$game"
  local target="$game_dir/$filename"
  local url="$GCS_ATARI_BASE_URL/$game/$filename"
  local expected_size
  local actual_size

  mkdir -p "$game_dir"
  expected_size="$(
    curl -fsSLI \
      --connect-timeout "$CURL_CONNECT_TIMEOUT" \
      --max-time 120 \
      "$url" |
      awk 'BEGIN { IGNORECASE=1 } /^content-length:/ { gsub("\r", "", $2); print $2; exit }'
  )"
  if [ -z "$expected_size" ]; then
    echo "Could not read Content-Length for $url" >&2
    return 1
  fi

  if [ -f "$target" ]; then
    actual_size="$(stat -c '%s' "$target")"
    if [ "$actual_size" = "$expected_size" ]; then
      echo "Raw shard already complete: $target ($actual_size bytes)"
      return 0
    fi
    if [ "$actual_size" -gt "$expected_size" ]; then
      mv "$target" "$target.bad.$(date +%Y%m%dT%H%M%S)"
    fi
  fi

  local attempt=1
  while true; do
    actual_size=0
    if [ -f "$target" ]; then
      actual_size="$(stat -c '%s' "$target")"
    fi
    echo "Downloading raw shard: $target attempt $attempt/$CURL_RETRIES ($actual_size/$expected_size bytes)"
    if curl -fL \
      --http1.1 \
      --continue-at - \
      --show-error \
      --no-progress-meter \
      --connect-timeout "$CURL_CONNECT_TIMEOUT" \
      --speed-limit "$CURL_SPEED_LIMIT" \
      --speed-time "$CURL_SPEED_TIME" \
      -o "$target" \
      "$url"; then
      break
    fi

    actual_size=0
    if [ -f "$target" ]; then
      actual_size="$(stat -c '%s' "$target")"
    fi
    if [ "$actual_size" = "$expected_size" ]; then
      break
    fi
    if [ "$attempt" -ge "$CURL_RETRIES" ]; then
      echo "Failed to download $target after $CURL_RETRIES attempts ($actual_size/$expected_size bytes)" >&2
      return 1
    fi
    attempt=$((attempt + 1))
    sleep 5
  done

  actual_size="$(stat -c '%s' "$target")"
  if [ "$actual_size" != "$expected_size" ]; then
    echo "Downloaded size mismatch for $target: got $actual_size, expected $expected_size" >&2
    return 1
  fi
}

download_raw_shards() {
  local game="$1"
  local num_shards="$2"

  echo "Downloading raw RLU Atari shards for $game run $TFDS_RUN to $RAW_INPUT_PREFIX/$game"
  export RAW_INPUT_PREFIX GCS_ATARI_BASE_URL CURL_RETRIES CURL_CONNECT_TIMEOUT CURL_SPEED_LIMIT CURL_SPEED_TIME
  export -f download_one_shard

  for ((shard = 0; shard < num_shards; shard++)); do
    printf '%05d\n' "$shard"
  done |
    xargs -r -I {} -P "$PARALLEL_DOWNLOADS" \
      bash -c 'download_one_shard "$1" "$2" "$3" "$4"' _ "$game" "$TFDS_RUN" "$num_shards" {}
}

for game in $GAMES; do
  num_shards="$(num_shards_for_game "$game")"
  if [ "$DOWNLOAD_RAW_SHARDS" = "1" ]; then
    download_raw_shards "$game" "$num_shards"
  fi

  echo "Downloading/preparing TFDS rlu_atari/${game}_run_${TFDS_RUN} to $TFDS_DATA_DIR"
  python - "$game" "$TFDS_RUN" "$TFDS_DATA_DIR" "$RAW_INPUT_PREFIX" "$USE_LOCAL_RAW_SHARDS" <<'PY'
import os
import sys

import tensorflow_datasets as tfds

game, run, data_dir, raw_input_prefix, use_local_raw_shards = sys.argv[1:]

if use_local_raw_shards == "1":
    from tensorflow_datasets.rl_unplugged.rlu_atari import rlu_atari

    rlu_atari.RluAtari._INPUT_FILE_PREFIX = os.path.abspath(raw_input_prefix)

builder = tfds.builder(f"rlu_atari/{game}_run_{run}", data_dir=data_dir)
print("TFDS builder:", builder.info.full_name)
print("TFDS data dir:", data_dir)
print("RLU input prefix:", builder.get_file_prefix())
builder.download_and_prepare()
print(builder.info)
PY
done
