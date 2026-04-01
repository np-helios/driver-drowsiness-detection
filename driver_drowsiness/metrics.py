from __future__ import annotations

from math import dist


def eye_aspect_ratio(eye) -> float:
    """Compute the eye aspect ratio from six eye landmarks."""
    vertical_a = dist(eye[1], eye[5])
    vertical_b = dist(eye[2], eye[4])
    horizontal = dist(eye[0], eye[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)
