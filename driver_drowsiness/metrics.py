from __future__ import annotations

from math import atan2, degrees, dist


def eye_aspect_ratio(eye) -> float:
    """Compute the eye aspect ratio from six eye landmarks."""
    vertical_a = dist(eye[1], eye[5])
    vertical_b = dist(eye[2], eye[4])
    horizontal = dist(eye[0], eye[3])
    return (vertical_a + vertical_b) / (2.0 * horizontal)


def mouth_aspect_ratio(mouth) -> float:
    """Compute mouth opening using an EAR-style ratio."""
    vertical_a = dist(mouth[1], mouth[7])
    vertical_b = dist(mouth[2], mouth[6])
    vertical_c = dist(mouth[3], mouth[5])
    horizontal = dist(mouth[0], mouth[4])
    return (vertical_a + vertical_b + vertical_c) / (3.0 * horizontal)


def head_roll_degrees(left_eye_corner, right_eye_corner) -> float:
    dx = right_eye_corner[0] - left_eye_corner[0]
    dy = right_eye_corner[1] - left_eye_corner[1]
    return degrees(atan2(dy, dx))


def head_pitch_proxy(nose_tip, chin, eye_center) -> float:
    """Approximate nodding/downward tilt using normalized vertical geometry."""
    face_height = max(dist(eye_center, chin), 1e-3)
    nose_drop = nose_tip[1] - eye_center[1]
    return nose_drop / face_height
