from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


MODEL_POINTS = np.array(
    [
        (0.0, 0.0, 0.0),
        (0.0, -63.6, -12.5),
        (-43.3, 32.7, -26.0),
        (43.3, 32.7, -26.0),
        (-28.9, -28.9, -24.1),
        (28.9, -28.9, -24.1),
    ],
    dtype=np.float64,
)


@dataclass(frozen=True)
class HeadPose:
    yaw: float
    pitch: float
    roll: float


def estimate_head_pose(
    frame_shape,
    nose_tip,
    chin,
    left_eye_corner,
    right_eye_corner,
    left_mouth_corner,
    right_mouth_corner,
) -> HeadPose:
    image_points = np.array(
        [
            nose_tip,
            chin,
            left_eye_corner,
            right_eye_corner,
            left_mouth_corner,
            right_mouth_corner,
        ],
        dtype=np.float64,
    )

    height, width = frame_shape[:2]
    focal_length = float(width)
    center = (width / 2.0, height / 2.0)
    camera_matrix = np.array(
        [
            [focal_length, 0.0, center[0]],
            [0.0, focal_length, center[1]],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    dist_coeffs = np.zeros((4, 1), dtype=np.float64)

    success, rotation_vector, _ = cv2.solvePnP(
        MODEL_POINTS,
        image_points,
        camera_matrix,
        dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE,
    )
    if not success:
        return HeadPose(yaw=0.0, pitch=0.0, roll=0.0)

    rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
    angles, *_ = cv2.RQDecomp3x3(rotation_matrix)
    pitch, yaw, roll = (float(angle) for angle in angles)
    return HeadPose(yaw=yaw, pitch=pitch, roll=roll)
