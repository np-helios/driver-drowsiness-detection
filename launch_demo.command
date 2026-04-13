#!/bin/zsh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"

if [ ! -d "$ROOT_DIR/.venv" ]; then
  "$ROOT_DIR/scripts/setup_macos.sh"
fi

"$ROOT_DIR/scripts/run_detector.sh" --driver-id exam_demo --baseline-seconds 20 --recalibrate
