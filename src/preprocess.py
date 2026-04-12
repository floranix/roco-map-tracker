from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils import AppConfig, load_image, resize_image


@dataclass
class MapBundle:
    color: np.ndarray
    gray: np.ndarray


class MapPreprocessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_map(self) -> MapBundle:
        color = load_image(self.config.map_path, grayscale=False)
        color = resize_image(color, self.config.resize_ratio)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        return MapBundle(color=color, gray=gray)

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        frame = self._preprocess_frame(frame)
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        return resize_image(gray, self.config.resize_ratio)

    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        mode = (self.config.frame_preprocess_mode or "none").strip().lower()
        if mode == "minimap_circle":
            return self._mask_minimap_frame(frame)
        return frame.copy()

    def _mask_minimap_frame(self, frame: np.ndarray) -> np.ndarray:
        height, width = frame.shape[:2]
        if height < 120 or width < 120:
            return frame.copy()

        circle = self._detect_main_circle(frame)
        if circle is None:
            center_x = width // 2
            center_y = height // 2
            radius = max(1, min(width, height) // 2 - self.config.minimap_outer_margin)
        else:
            center_x, center_y, radius = circle
            radius = max(1, radius - self.config.minimap_outer_margin)

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask, (center_x, center_y), radius, 255, -1)

        center_mask_radius = max(1, int(round(radius * self.config.minimap_center_mask_ratio)))
        cv2.circle(mask, (center_x, center_y), center_mask_radius, 0, -1)

        icon_x = int(round(center_x + radius * self.config.minimap_icon_offset_x_ratio))
        icon_y = int(round(center_y + radius * self.config.minimap_icon_offset_y_ratio))
        icon_radius = max(1, int(round(radius * self.config.minimap_icon_mask_ratio)))
        cv2.circle(mask, (icon_x, icon_y), icon_radius, 0, -1)

        masked = frame.copy()
        masked[mask == 0] = 0
        return masked

    @staticmethod
    def _detect_main_circle(frame: np.ndarray) -> tuple[int, int, int] | None:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        blurred = cv2.GaussianBlur(gray, (9, 9), 1.5)
        min_radius = max(40, min(gray.shape[:2]) // 3)
        max_radius = max(min_radius + 1, min(gray.shape[:2]) // 2 + 12)
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=min(gray.shape[:2]) // 3,
            param1=120,
            param2=30,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is None or len(circles[0]) == 0:
            return None

        circle = max(circles[0], key=lambda item: item[2])
        return int(round(circle[0])), int(round(circle[1])), int(round(circle[2]))
