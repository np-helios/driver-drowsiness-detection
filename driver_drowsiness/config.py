from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional


BASE_DIR = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class DetectorConfig:
    driver_id: str = "default-driver"
    force_recalibration: bool = False
    face_landmarker_task_path: Path = BASE_DIR / "models" / "face_landmarker.task"
    init_alarm_path: Path = BASE_DIR / "init_sound.mp3"
    short_alarm_path: Path = BASE_DIR / "short_alarm.mp3"
    normal_alarm_path: Path = BASE_DIR / "nomal_alarm.wav"
    power_alarm_path: Path = BASE_DIR / "power_alarm.wav"
    frame_width: int = 400
    camera_source: int = 0
    warmup_seconds: float = 1.0
    open_eye_delay_seconds: float = 5.0
    close_eye_delay_seconds: float = 5.0
    calibration_samples: int = 7
    calibration_interval_seconds: float = 1.0
    ear_consecutive_frames: int = 20
    min_closed_duration_seconds: float = 1.1
    alarm_cooldown_seconds: float = 2.5
    min_yawn_duration_seconds: float = 1.4
    light_blur_kernel: int = 99
    eye_scale_factor: float = 500.0
    personalized_calibration_seconds: float = 90.0
    baseline_path: Path = BASE_DIR / "profiles" / "default-driver.json"
    event_log_path: Optional[Path] = BASE_DIR / "logs" / "events.jsonl"
    session_summary_path: Optional[Path] = BASE_DIR / "logs" / "session_summary.json"
