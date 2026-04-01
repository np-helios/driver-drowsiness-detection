from __future__ import annotations

import subprocess
from pathlib import Path


def sound_alarm(path: Path) -> None:
    subprocess.run(["afplay", str(path)], check=False)
