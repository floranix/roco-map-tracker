from __future__ import annotations

from typing import Optional

import cv2
import numpy as np


class PositionKalmanFilter:
    def __init__(self) -> None:
        self.filter = cv2.KalmanFilter(4, 2)
        self.filter.measurementMatrix = np.array(
            [[1, 0, 0, 0], [0, 1, 0, 0]],
            dtype=np.float32,
        )
        self.filter.transitionMatrix = np.array(
            [[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float32,
        )
        self.filter.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.filter.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.5
        self.filter.errorCovPost = np.eye(4, dtype=np.float32)
        self.initialized = False

    def predict(self) -> Optional[tuple[float, float]]:
        if not self.initialized:
            return None
        state = self.filter.predict()
        return float(state[0, 0]), float(state[1, 0])

    def correct(self, x: float, y: float) -> tuple[float, float]:
        if not self.initialized:
            self.reset(x, y)
            return x, y

        measurement = np.array([[x], [y]], dtype=np.float32)
        state = self.filter.correct(measurement)
        return float(state[0, 0]), float(state[1, 0])

    def reset(self, x: float, y: float) -> None:
        state = np.array([[x], [y], [0.0], [0.0]], dtype=np.float32)
        self.filter.statePost = state.copy()
        self.filter.statePre = state.copy()
        self.initialized = True
