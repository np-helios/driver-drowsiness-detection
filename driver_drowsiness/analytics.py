from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass
class SessionAnalytics:
    frames_processed: int = 0
    face_detections: int = 0
    blink_events: int = 0
    yawn_events: int = 0
    alerts_triggered: int = 0
    break_recommendations: int = 0
    max_fatigue_score: float = 0.0
    total_ear: float = 0.0
    total_mar: float = 0.0
    total_head_yaw: float = 0.0
    total_head_pitch: float = 0.0
    scored_frames: int = 0

    def add_frame(self, ear: float, mar: float, head_yaw: float, head_pitch: float, face_detected: bool) -> None:
        self.frames_processed += 1
        if face_detected:
            self.face_detections += 1
            self.total_ear += ear
            self.total_mar += mar
            self.total_head_yaw += head_yaw
            self.total_head_pitch += head_pitch
            self.scored_frames += 1

    def record_score(self, score: float) -> None:
        self.max_fatigue_score = max(self.max_fatigue_score, score)

    def summary(self) -> dict:
        denom = max(self.scored_frames, 1)
        return {
            "frames_processed": self.frames_processed,
            "face_detections": self.face_detections,
            "blink_events": self.blink_events,
            "yawn_events": self.yawn_events,
            "alerts_triggered": self.alerts_triggered,
            "break_recommendations": self.break_recommendations,
            "max_fatigue_score": round(self.max_fatigue_score, 3),
            "average_ear": round(self.total_ear / denom, 3),
            "average_mar": round(self.total_mar / denom, 3),
            "average_head_yaw": round(self.total_head_yaw / denom, 3),
            "average_head_pitch": round(self.total_head_pitch / denom, 3),
        }


def save_session_summary(path: Path | None, summary: dict) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
