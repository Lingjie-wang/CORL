#!/usr/bin/env bash
# Install Atari 2600 ROMs into atari-py so online evaluation can launch the
# emulator. pip-installed atari-py ships only tetris.bin, so without this step
# scripts/run_atari_*_wlj.sh crash at eval time with "ROM is missing for <game>".
#
# Idempotent: if the ROMs are already present it exits early. Safe to re-run and
# safe to run on a fresh server after `pip install -r requirements/requirements_atari.txt`.
#
# Usage:
#   CONDA_ENV=corl scripts/setup_atari_roms_wlj.sh
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

# Games needed by the run scripts; extend if you add more.
CHECK_GAMES="${CHECK_GAMES:-seaquest breakout qbert pong}"

if python - "$CHECK_GAMES" <<'PY'
import sys
import atari_py
missing = []
for game in sys.argv[1].split():
    try:
        atari_py.get_game_path(game)
    except Exception:
        missing.append(game)
sys.exit(1 if missing else 0)
PY
then
  echo "Atari ROMs already installed for: $CHECK_GAMES"
  exit 0
fi

echo "Some ROMs are missing; installing via AutoROM..."
python -m pip install "autorom[accept-rom-license]"
AutoROM --accept-license

AUTOROM_DIR="$(python -c 'import AutoROM, os; print(os.path.join(os.path.dirname(AutoROM.__file__), "roms"))')"
echo "Importing ROMs from $AUTOROM_DIR into atari-py..."
python -m atari_py.import_roms "$AUTOROM_DIR"

echo "Verifying..."
python - "$CHECK_GAMES" <<'PY'
import sys
import atari_py
for game in sys.argv[1].split():
    atari_py.get_game_path(game)
    print(f"  OK: {game}")
print("All requested Atari ROMs installed.")
PY
