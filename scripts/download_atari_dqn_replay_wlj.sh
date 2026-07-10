#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

GAMES="${GAMES:-Breakout Seaquest Qbert Pong}"
ATARI_DATA_DIR="${ATARI_DATA_DIR:-$REPO_ROOT/outputs/atari/dqn_replay}"
FORCE_DOWNLOAD="${FORCE_DOWNLOAD:-0}"
DOWNLOAD_TOOL="${DOWNLOAD_TOOL:-auto}"

mkdir -p "$ATARI_DATA_DIR"

print_auth_help() {
  cat >&2 <<'EOF'

Atari DQN Replay download failed because Google Cloud Storage denied access.
The original Decision Transformer command uses:
  gsutil -m cp -R gs://atari-replay-datasets/dqn/[GAME] [DATA_DIR]

If this bucket is not anonymously accessible from your environment, authenticate first:
  gcloud auth login
  gcloud auth application-default login

If you installed standalone gsutil with pip and do not have Cloud SDK credentials,
configure credentials for gsutil:
  gsutil config

Then rerun this script. If the data is already downloaded elsewhere, skip download
and point the runners at it:
  DOWNLOAD_DATA=0 ATARI_DATA_DIR=/path/to/dqn_replay scripts/run_atari_dt_wlj.sh
EOF
}

download_with_gsutil() {
  local game="$1"
  gsutil -m cp -n -R "gs://atari-replay-datasets/dqn/$game" "$ATARI_DATA_DIR"
}

download_with_gcloud() {
  local game="$1"
  gcloud storage cp --recursive --no-clobber \
    "gs://atari-replay-datasets/dqn/$game" "$ATARI_DATA_DIR"
}

for game in $GAMES; do
  game_dir="$ATARI_DATA_DIR/$game"
  replay_dir="$game_dir/1/replay_logs"
  existing_file="$(find "$replay_dir" -type f -print -quit 2>/dev/null || true)"
  if [ "$FORCE_DOWNLOAD" != "1" ] && [ -n "$existing_file" ]; then
    echo "Atari dataset already exists for $game: $game_dir"
    continue
  fi

  tool="$DOWNLOAD_TOOL"
  if [ "$tool" = "auto" ]; then
    if command -v gcloud >/dev/null 2>&1; then
      tool="gcloud"
    elif command -v gsutil >/dev/null 2>&1; then
      tool="gsutil"
    else
      echo "Google Cloud CLI is required to download Atari DQN Replay datasets." >&2
      echo "Install Google Cloud SDK, or install gsutil and authenticate it." >&2
      exit 1
    fi
  fi

  echo "Downloading Atari DQN Replay dataset for $game to $ATARI_DATA_DIR with $tool"
  if [ "$tool" = "gcloud" ]; then
    if ! command -v gcloud >/dev/null 2>&1; then
      echo "DOWNLOAD_TOOL=gcloud was requested, but gcloud is not installed." >&2
      exit 1
    fi
    if ! download_with_gcloud "$game"; then
      print_auth_help
      exit 1
    fi
  elif [ "$tool" = "gsutil" ]; then
    if ! command -v gsutil >/dev/null 2>&1; then
      echo "DOWNLOAD_TOOL=gsutil was requested, but gsutil is not installed." >&2
      exit 1
    fi
    if ! download_with_gsutil "$game"; then
      print_auth_help
      exit 1
    fi
  else
    echo "Unsupported DOWNLOAD_TOOL=$tool. Use auto, gcloud, or gsutil." >&2
    exit 1
  fi
done
