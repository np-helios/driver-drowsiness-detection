#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  if command -v uv >/dev/null 2>&1; then
    uv venv --seed --python 3.11 "$VENV_DIR"
  elif command -v python3.11 >/dev/null 2>&1; then
    python3.11 -m venv "$VENV_DIR"
  else
    echo "Python 3.11 or uv is required to set up this project."
    exit 1
  fi
fi

source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$ROOT_DIR/requirements.txt"
python -m pip install -e "$ROOT_DIR"

echo
echo "Setup complete."
echo "Run the detector with:"
echo "  $ROOT_DIR/scripts/run_detector.sh --driver-id demo --baseline-seconds 20 --recalibrate"
