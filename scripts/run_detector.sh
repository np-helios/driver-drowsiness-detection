#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV_DIR="$ROOT_DIR/.venv"

if [ ! -d "$VENV_DIR" ]; then
  echo "Virtual environment not found."
  echo "Run $ROOT_DIR/scripts/setup_macos.sh first."
  exit 1
fi

source "$VENV_DIR/bin/activate"
cd "$ROOT_DIR"

python drowsiness_detector.py "$@"
