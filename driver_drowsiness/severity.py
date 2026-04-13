from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineProfile, RollingBehavior, z_score


@dataclass(frozen=True)
class SeverityResult:
    level: int
    alarm_name: str
    rationale: str
    score: float
    break_recommended: bool = False


def classify_deviation(
    baseline: BaselineProfile,
    behavior: RollingBehavior,
    current_ear: float,
    current_mar: float,
    current_head_yaw: float,
    current_head_roll: float,
    current_head_pitch: float,
) -> SeverityResult:
    """
    Personalized severity scoring based on deviation from a driver's baseline.
    Lower EAR and longer closures relative to baseline increase the risk score.
    """
    if not behavior.has_enough_signal:
        return SeverityResult(level=2, alarm_name="short", rationale="insufficient_behavior_signal", score=0.0)

    ear_drop = max(0.0, -z_score(current_ear, baseline.open_ear_mean, baseline.open_ear_std, floor=2.5))
    mar_risk = max(0.0, z_score(current_mar, baseline.mar_mean, baseline.mar_std, floor=0.03))
    head_yaw_risk = abs(_angular_z_score(current_head_yaw, baseline.head_yaw_mean, baseline.head_yaw_std, floor=8.0))
    head_roll_risk = abs(_angular_z_score(current_head_roll, baseline.head_roll_mean, baseline.head_roll_std, floor=6.0))
    head_pitch_risk = abs(_angular_z_score(current_head_pitch, baseline.head_pitch_mean, baseline.head_pitch_std, floor=10.0))
    blink_duration_risk = max(
        0.0,
        z_score(
            behavior.blink_durations[-1],
            baseline.blink_duration_mean,
            baseline.blink_duration_std,
            floor=0.2,
        ),
    )
    open_interval_risk = max(
        0.0,
        -z_score(
            behavior.open_intervals[-1],
            baseline.open_interval_mean,
            baseline.open_interval_std,
            floor=1.0,
        ),
    )
    yawn_risk = 0.0
    if behavior.yawn_durations:
        yawn_risk = max(
            0.0,
            z_score(
                behavior.yawn_durations[-1],
                baseline.yawn_duration_mean,
                baseline.yawn_duration_std,
                floor=0.4,
            ),
        )

    score = (
        ear_drop * 0.28
        + blink_duration_risk * 0.2
        + open_interval_risk * 0.1
        + mar_risk * 0.18
        + yawn_risk * 0.14
        + min(head_yaw_risk, 2.0) * 0.05
        + min(head_pitch_risk, 2.0) * 0.05
        + min(head_roll_risk, 2.0) * 0.02
    )

    if score >= 3.2:
        return SeverityResult(
            level=0,
            alarm_name="power",
            rationale="strong deviation from personal baseline",
            score=round(score, 3),
            break_recommended=True,
        )
    if score >= 1.8:
        return SeverityResult(
            level=1,
            alarm_name="normal",
            rationale="moderate deviation from personal baseline",
            score=round(score, 3),
            break_recommended=score >= 2.6,
        )
    return SeverityResult(
        level=2,
        alarm_name="short",
        rationale="mild deviation from personal baseline",
        score=round(score, 3),
    )


def _angular_delta(value: float, baseline_mean: float) -> float:
    return ((value - baseline_mean + 180.0) % 360.0) - 180.0


def _angular_z_score(value: float, baseline_mean: float, baseline_std: float, floor: float) -> float:
    return _angular_delta(value, baseline_mean) / max(baseline_std, floor)
