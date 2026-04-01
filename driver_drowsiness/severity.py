from __future__ import annotations

from dataclasses import dataclass

from .baseline import BaselineProfile, RollingBehavior, z_score


@dataclass(frozen=True)
class SeverityResult:
    level: int
    alarm_name: str
    rationale: str
    score: float


def classify_deviation(
    baseline: BaselineProfile,
    behavior: RollingBehavior,
    current_ear: float,
) -> SeverityResult:
    """
    Personalized severity scoring based on deviation from a driver's baseline.
    Lower EAR and longer closures relative to baseline increase the risk score.
    """
    if not behavior.has_enough_signal:
        return SeverityResult(level=2, alarm_name="short", rationale="insufficient_behavior_signal", score=0.0)

    ear_drop = max(0.0, -z_score(current_ear, baseline.open_ear_mean, baseline.open_ear_std, floor=2.5))
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

    score = ear_drop * 0.45 + blink_duration_risk * 0.4 + open_interval_risk * 0.15

    if score >= 3.2:
        return SeverityResult(
            level=0,
            alarm_name="power",
            rationale="strong deviation from personal baseline",
            score=round(score, 3),
        )
    if score >= 1.8:
        return SeverityResult(
            level=1,
            alarm_name="normal",
            rationale="moderate deviation from personal baseline",
            score=round(score, 3),
        )
    return SeverityResult(
        level=2,
        alarm_name="short",
        rationale="mild deviation from personal baseline",
        score=round(score, 3),
    )
