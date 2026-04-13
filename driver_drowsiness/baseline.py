from __future__ import annotations

import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, pstdev


@dataclass(frozen=True)
class BaselineProfile:
    driver_id: str
    open_ear_mean: float
    open_ear_std: float
    mar_mean: float
    mar_std: float
    head_yaw_mean: float
    head_yaw_std: float
    head_roll_mean: float
    head_roll_std: float
    head_pitch_mean: float
    head_pitch_std: float
    blink_duration_mean: float
    blink_duration_std: float
    open_interval_mean: float
    open_interval_std: float
    yawn_duration_mean: float
    yawn_duration_std: float
    sample_count: int


class BaselineStore:
    def __init__(self, path: Path):
        self.path = path

    def load(self) -> BaselineProfile | None:
        if not self.path.exists():
            return None
        with self.path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        defaults = {
            "mar_mean": 0.0,
            "mar_std": 0.0,
            "head_yaw_mean": 0.0,
            "head_yaw_std": 0.0,
            "head_roll_mean": 0.0,
            "head_roll_std": 0.0,
            "head_pitch_mean": 0.0,
            "head_pitch_std": 0.0,
            "yawn_duration_mean": 0.0,
            "yawn_duration_std": 0.0,
        }
        defaults.update(payload)
        return BaselineProfile(**defaults)

    def save(self, profile: BaselineProfile) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as handle:
            json.dump(asdict(profile), handle, indent=2)


class BaselineCalibrator:
    def __init__(self) -> None:
        self._open_ear_samples: list[float] = []
        self._mar_samples: list[float] = []
        self._head_yaw_samples: list[float] = []
        self._head_roll_samples: list[float] = []
        self._head_pitch_samples: list[float] = []
        self._blink_duration_samples: list[float] = []
        self._open_interval_samples: list[float] = []
        self._yawn_duration_samples: list[float] = []

    def add_open_ear(self, value: float) -> None:
        self._open_ear_samples.append(value)

    def add_mar(self, value: float) -> None:
        self._mar_samples.append(value)

    def add_head_pose(self, yaw: float, roll: float, pitch: float) -> None:
        self._head_yaw_samples.append(yaw)
        self._head_roll_samples.append(roll)
        self._head_pitch_samples.append(pitch)

    def add_blink_metrics(self, blink_duration: float, open_interval: float) -> None:
        self._blink_duration_samples.append(blink_duration)
        self._open_interval_samples.append(open_interval)

    def add_yawn_duration(self, yawn_duration: float) -> None:
        self._yawn_duration_samples.append(yawn_duration)

    def build(self, driver_id: str) -> BaselineProfile:
        return BaselineProfile(
            driver_id=driver_id,
            open_ear_mean=_safe_mean(self._open_ear_samples),
            open_ear_std=_safe_std(self._open_ear_samples),
            mar_mean=_safe_mean(self._mar_samples),
            mar_std=_safe_std(self._mar_samples),
            head_yaw_mean=_safe_mean(self._head_yaw_samples),
            head_yaw_std=_safe_std(self._head_yaw_samples),
            head_roll_mean=_safe_mean(self._head_roll_samples),
            head_roll_std=_safe_std(self._head_roll_samples),
            head_pitch_mean=_safe_mean(self._head_pitch_samples),
            head_pitch_std=_safe_std(self._head_pitch_samples),
            blink_duration_mean=_safe_mean(self._blink_duration_samples),
            blink_duration_std=_safe_std(self._blink_duration_samples),
            open_interval_mean=_safe_mean(self._open_interval_samples),
            open_interval_std=_safe_std(self._open_interval_samples),
            yawn_duration_mean=_safe_mean(self._yawn_duration_samples),
            yawn_duration_std=_safe_std(self._yawn_duration_samples),
            sample_count=len(self._open_ear_samples),
        )


class RollingBehavior:
    def __init__(self, maxlen: int = 20) -> None:
        self.open_ears: deque[float] = deque(maxlen=maxlen)
        self.mars: deque[float] = deque(maxlen=maxlen)
        self.head_yaws: deque[float] = deque(maxlen=maxlen)
        self.head_rolls: deque[float] = deque(maxlen=maxlen)
        self.head_pitches: deque[float] = deque(maxlen=maxlen)
        self.blink_durations: deque[float] = deque(maxlen=maxlen)
        self.open_intervals: deque[float] = deque(maxlen=maxlen)
        self.yawn_durations: deque[float] = deque(maxlen=maxlen)

    def add_open_ear(self, value: float) -> None:
        self.open_ears.append(value)

    def add_mar(self, value: float) -> None:
        self.mars.append(value)

    def add_head_pose(self, yaw: float, roll: float, pitch: float) -> None:
        self.head_yaws.append(yaw)
        self.head_rolls.append(roll)
        self.head_pitches.append(pitch)

    def add_blink_metrics(self, blink_duration: float, open_interval: float) -> None:
        self.blink_durations.append(blink_duration)
        self.open_intervals.append(open_interval)

    def add_yawn_duration(self, yawn_duration: float) -> None:
        self.yawn_durations.append(yawn_duration)

    @property
    def has_enough_signal(self) -> bool:
        return bool(
            self.open_ears
            and self.mars
            and self.head_yaws
            and self.head_rolls
            and self.head_pitches
            and self.blink_durations
            and self.open_intervals
        )


def z_score(value: float, baseline_mean: float, baseline_std: float, floor: float = 1e-3) -> float:
    return (value - baseline_mean) / max(baseline_std, floor)


def _safe_mean(values: list[float]) -> float:
    return mean(values) if values else 0.0


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return pstdev(values)
