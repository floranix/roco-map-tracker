from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from src.utils import LocalizationResult


@dataclass
class TrackerMemory:
    last_result: Optional[LocalizationResult] = None
    last_frame_gray: Optional[np.ndarray] = None
    predicted_center: Optional[tuple[float, float]] = None
    lost_frames: int = 0


class LocalTracker:
    def __init__(
        self,
        roi_expand_pixels: int,
        max_lost_frames: int,
        use_optical_flow: bool = True,
        motion_gate_pixels: float = 210.0,
        motion_gate_per_lost_frame: float = 45.0,
    ) -> None:
        self.roi_expand_pixels = roi_expand_pixels
        self.max_lost_frames = max_lost_frames
        self.use_optical_flow = use_optical_flow
        self.motion_gate_pixels = float(motion_gate_pixels)
        self.motion_gate_per_lost_frame = float(motion_gate_per_lost_frame)
        self.memory = TrackerMemory()

    def build_search_region(
        self,
        current_frame_gray: np.ndarray,
        map_shape: tuple[int, int],
    ) -> Optional[tuple[int, int, int, int]]:
        if self.memory.last_result is None:
            return None

        predicted_x = self.memory.last_result.x
        predicted_y = self.memory.last_result.y
        delta = self._estimate_optical_flow_delta(current_frame_gray)
        if delta is not None:
            predicted_x += delta[0]
            predicted_y += delta[1]
        self.memory.predicted_center = (predicted_x, predicted_y)

        bbox_width, bbox_height = self._result_extent(self.memory.last_result, current_frame_gray.shape)
        expansion = self.roi_expand_pixels + self.memory.lost_frames * 40
        half_width = max(current_frame_gray.shape[1], bbox_width) / 2 + expansion
        half_height = max(current_frame_gray.shape[0], bbox_height) / 2 + expansion

        map_height, map_width = map_shape
        x0 = max(0, int(round(predicted_x - half_width)))
        y0 = max(0, int(round(predicted_y - half_height)))
        x1 = min(map_width, int(round(predicted_x + half_width)))
        y1 = min(map_height, int(round(predicted_y + half_height)))

        if x1 - x0 < current_frame_gray.shape[1] or y1 - y0 < current_frame_gray.shape[0]:
            return None
        return x0, y0, x1, y1

    def register_success(self, frame_gray: np.ndarray, result: LocalizationResult) -> None:
        self.memory.last_result = result
        self.memory.last_frame_gray = frame_gray.copy()
        self.memory.predicted_center = (result.x, result.y)
        self.memory.lost_frames = 0

    def register_failure(self) -> None:
        self.memory.lost_frames += 1

    def is_lost(self) -> bool:
        return self.memory.lost_frames >= self.max_lost_frames

    def is_result_plausible(
        self,
        result: Optional[LocalizationResult],
        frame_shape: tuple[int, int],
        strict: bool = True,
    ) -> bool:
        if result is None:
            return False
        if self.memory.last_result is None:
            return True
        if any(math.isnan(value) for value in (result.x, result.y)):
            return False

        anchor = self.memory.predicted_center or (self.memory.last_result.x, self.memory.last_result.y)
        if any(math.isnan(value) for value in anchor):
            return True

        distance = float(np.hypot(result.x - anchor[0], result.y - anchor[1]))
        if distance > self.motion_limit(frame_shape, strict=strict):
            return False

        if strict:
            previous_width, previous_height = self._result_extent(self.memory.last_result, frame_shape)
            current_width, current_height = self._result_extent(result, frame_shape)
            width_ratio = max(current_width / max(previous_width, 1.0), previous_width / max(current_width, 1.0))
            height_ratio = max(current_height / max(previous_height, 1.0), previous_height / max(current_height, 1.0))
            if width_ratio > 1.8 or height_ratio > 1.8:
                return False
        return True

    def motion_limit(self, frame_shape: tuple[int, int], strict: bool = True) -> float:
        frame_height, frame_width = frame_shape[:2]
        diagonal = float(np.hypot(frame_width, frame_height))
        base = max(self.motion_gate_pixels, diagonal * (0.78 if strict else 1.05))
        return base + self.memory.lost_frames * self.motion_gate_per_lost_frame

    def _estimate_optical_flow_delta(self, current_frame_gray: np.ndarray) -> Optional[tuple[float, float]]:
        if not self.use_optical_flow:
            return None

        previous_frame = self.memory.last_frame_gray
        if previous_frame is None or previous_frame.shape != current_frame_gray.shape:
            return None

        previous_points = cv2.goodFeaturesToTrack(
            previous_frame,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=7,
            blockSize=7,
        )
        if previous_points is None or len(previous_points) < 5:
            return None

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_frame,
            current_frame_gray,
            previous_points,
            None,
        )
        if current_points is None or status is None:
            return None

        valid = status.reshape(-1) == 1
        if valid.sum() < 5:
            return None

        flow = current_points[valid] - previous_points[valid]
        median_flow = np.median(flow.reshape(-1, 2), axis=0)
        return -float(median_flow[0]), -float(median_flow[1])

    @staticmethod
    def _result_extent(
        result: LocalizationResult,
        frame_shape: tuple[int, int],
    ) -> tuple[float, float]:
        if result.bbox is not None:
            x0, y0, x1, y1 = result.bbox
            return float(max(1, x1 - x0)), float(max(1, y1 - y0))
        if result.corners:
            corners = np.array(result.corners, dtype=np.float32)
            return (
                float(max(1.0, corners[:, 0].max() - corners[:, 0].min())),
                float(max(1.0, corners[:, 1].max() - corners[:, 1].min())),
            )
        return float(frame_shape[1]), float(frame_shape[0])
