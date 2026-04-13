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

python -m pip install pyinstaller

pyinstaller \
  --noconfirm \
  --windowed \
  --name "Driver Drowsiness Detector" \
  --add-data "models:models" \
  --add-data "profiles:profiles" \
  --add-data "init_sound.mp3:." \
  --add-data "short_alarm.mp3:." \
  --add-data "nomal_alarm.wav:." \
  --add-data "power_alarm.wav:." \
  drowsiness_detector.py

echo
echo "Build complete."
echo "App bundle: $ROOT_DIR/dist/Driver Drowsiness Detector.app"
