from __future__ import annotations

import cv2
import numpy as np


class DrowsinessKNN:
    def __init__(self, sample_size: int = 25, neighbors: int = 5, seed: int = 9):
        self.sample_size = sample_size
        self.neighbors = neighbors
        self._rng = np.random.default_rng(seed)
        self._knn = cv2.ml.KNearest_create()
        self.train()

    def train(self) -> None:
        train_data = self.generate_data(self.sample_size)
        labels = self.classify_label(train_data)
        self._knn.train(train_data, cv2.ml.ROW_SAMPLE, labels)

    def predict(self, opened_eyes_time: float, closing_time: float) -> int:
        sample = np.array([[opened_eyes_time, closing_time]], dtype=np.float32)
        _, results, _, _ = self._knn.findNearest(sample, self.neighbors)
        return int(results[0][0])

    def generate_data(self, num_samples: int, num_features: int = 2) -> np.ndarray:
        data = self._rng.integers(0, 40, size=(num_samples, num_features))
        return data.astype(np.float32)

    @staticmethod
    def classify_label(train_data: np.ndarray) -> np.ndarray:
        labels = []
        for opened_eyes_time, closing_time in train_data:
            if closing_time < opened_eyes_time - 15:
                labels.append(2)
            elif closing_time >= (opened_eyes_time / 2 + 15):
                labels.append(0)
            else:
                labels.append(1)
        return np.array(labels)
