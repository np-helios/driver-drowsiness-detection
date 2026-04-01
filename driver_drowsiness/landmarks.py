from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from urllib.request import urlretrieve

import cv2
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision


# MediaPipe Face Mesh landmark indices for EAR-style eye polygons.
LEFT_EYE_INDICES = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_INDICES = [362, 385, 387, 263, 373, 380]
DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
    "face_landmarker/float16/latest/face_landmarker.task"
)


@dataclass(frozen=True)
class EyeLandmarks:
    left_eye: list[tuple[float, float]]
    right_eye: list[tuple[float, float]]


class FaceMeshLandmarks:
    def __init__(
        self,
        model_path: Path,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
    ) -> None:
        self.model_path = model_path
        self._ensure_model()
        options = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(self.model_path)),
            running_mode=vision.RunningMode.IMAGE,
            num_faces=max_num_faces,
            min_face_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            output_face_blendshapes=False,
            output_facial_transformation_matrixes=False,
        )
        self._face_mesh = vision.FaceLandmarker.create_from_options(options)

    def detect(self, frame_bgr) -> EyeLandmarks | None:
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        results = self._face_mesh.detect(image)
        if not results.face_landmarks:
            return None

        face_landmarks = results.face_landmarks[0]
        height, width = frame_bgr.shape[:2]
        left_eye = [self._point(face_landmarks[index], width, height) for index in LEFT_EYE_INDICES]
        right_eye = [self._point(face_landmarks[index], width, height) for index in RIGHT_EYE_INDICES]
        return EyeLandmarks(left_eye=left_eye, right_eye=right_eye)

    def close(self) -> None:
        self._face_mesh.close()

    def _ensure_model(self) -> None:
        if self.model_path.exists():
            return
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading MediaPipe face landmarker model to {self.model_path} ...")
        urlretrieve(DEFAULT_MODEL_URL, self.model_path)

    @staticmethod
    def _point(landmark, width: int, height: int) -> tuple[float, float]:
        return (landmark.x * width, landmark.y * height)
