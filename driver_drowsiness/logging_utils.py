from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class EventLogger:
    def __init__(self, path: Path | None):
        self.path = path

    def log(self, event_type: str, **payload: Any) -> None:
        if self.path is None:
            return

        self.path.parent.mkdir(parents=True, exist_ok=True)
        normalized_payload = {}
        for key, value in payload.items():
            if is_dataclass(value):
                normalized_payload[key] = asdict(value)
            else:
                normalized_payload[key] = value

        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "event_type": event_type,
            **normalized_payload,
        }

        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")
