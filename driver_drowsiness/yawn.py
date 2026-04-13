from __future__ import annotations

from .baseline import BaselineProfile


def mar_activation_threshold(mar_mean: float, mar_std: float) -> float:
    return mar_mean + max(mar_std * 1.8, mar_mean * 0.55, 0.012)


def mar_confirmation_threshold(mar_mean: float, mar_std: float) -> float:
    return mar_mean + max(mar_std * 3.0, mar_mean * 1.1, 0.02)


def is_confirmed_yawn(
    *,
    duration_seconds: float,
    peak_mar: float,
    baseline_profile: BaselineProfile | None,
    min_duration_seconds: float,
) -> bool:
    if duration_seconds < min_duration_seconds:
        return False
    if baseline_profile is None:
        return peak_mar >= 0.045
    return peak_mar >= mar_confirmation_threshold(
        baseline_profile.mar_mean,
        baseline_profile.mar_std,
    )
