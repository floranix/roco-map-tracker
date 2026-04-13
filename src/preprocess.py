from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from src.utils import AppConfig, load_image, resize_image


@dataclass
class MapBundle:
    color: np.ndarray
    gray: np.ndarray


@dataclass
class PreparedFrame:
    gray: np.ndarray
    feature_mask: np.ndarray | None = None
    content_mask: np.ndarray | None = None


class MapPreprocessor:
    def __init__(self, config: AppConfig) -> None:
        self.config = config

    def load_map(self) -> MapBundle:
        color = load_image(self.config.map_path, grayscale=False)
        color = resize_image(color, self.config.resize_ratio)
        gray = cv2.cvtColor(color, cv2.COLOR_BGR2GRAY)
        return MapBundle(color=color, gray=gray)

    def prepare_frame(self, frame: np.ndarray) -> np.ndarray:
        return self.prepare_frame_bundle(frame).gray

    def prepare_frame_bundle(self, frame: np.ndarray) -> PreparedFrame:
        frame, feature_mask, content_mask = self._preprocess_frame(frame)
        if frame.ndim == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()
        gray = resize_image(gray, self.config.resize_ratio)
        feature_mask = self._resize_mask(feature_mask, self.config.resize_ratio)
        content_mask = self._resize_mask(content_mask, self.config.resize_ratio)
        return PreparedFrame(gray=gray, feature_mask=feature_mask, content_mask=content_mask)

    def _preprocess_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        mode = (self.config.frame_preprocess_mode or "none").strip().lower()
        if mode == "minimap_circle":
            return self._prepare_minimap_frame(frame)
        return frame.copy(), None, None

    def _prepare_minimap_frame(
        self,
        frame: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
        height, width = frame.shape[:2]
        if height < 120 or width < 120:
            return frame.copy(), None, None

        circle = self._detect_main_circle(frame)
        center_x = width // 2
        center_y = height // 2
        radius = max(1, min(width, height) // 2 - self.config.minimap_outer_margin)

        if circle is None:
            content_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(content_mask, (center_x, center_y), radius, 255, -1)
        else:
            center_x, center_y, radius = circle
            radius = max(1, radius - self.config.minimap_outer_margin)
            content_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(content_mask, (center_x, center_y), radius, 255, -1)

        center_mask_radius = max(1, int(round(radius * self.config.minimap_center_mask_ratio)))
        cv2.circle(content_mask, (center_x, center_y), center_mask_radius, 0, -1)

        if circle is not None:
            icon_x = int(round(center_x + radius * self.config.minimap_icon_offset_x_ratio))
            icon_y = int(round(center_y + radius * self.config.minimap_icon_offset_y_ratio))
            icon_radius = max(1, int(round(radius * self.config.minimap_icon_mask_ratio)))
            cv2.circle(content_mask, (icon_x, icon_y), icon_radius, 0, -1)

        self._mask_overlay_components(
            frame=frame,
            mask=content_mask,
            center_x=center_x,
            center_y=center_y,
            reference_radius=radius,
        )

        masked = frame.copy()
        masked[content_mask == 0] = 0
        feature_mask = self._build_feature_mask(content_mask)
        return masked, feature_mask, content_mask

    def _mask_overlay_components(
        self,
        frame: np.ndarray,
        mask: np.ndarray,
        center_x: int,
        center_y: int,
        reference_radius: int,
    ) -> None:
        if frame.ndim != 3:
            return

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        saturation_mask = ((hsv[:, :, 1] >= 70) & (hsv[:, :, 2] >= 90)).astype(np.uint8) * 255
        saturation_mask = cv2.morphologyEx(
            saturation_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), dtype=np.uint8),
        )

        label_count, labels, stats, centroids = cv2.connectedComponentsWithStats(saturation_mask, 8)
        frame_area = frame.shape[0] * frame.shape[1]
        min_area = max(18, int(round(frame_area * 0.0007)))
        max_area = max(min_area + 1, int(round(frame_area * 0.04)))
        max_width = max(12, int(round(frame.shape[1] * 0.35)))
        max_height = max(12, int(round(frame.shape[0] * 0.35)))
        edge_band = max(14, int(round(min(frame.shape[:2]) * 0.12)))
        center_distance_limit = max(12.0, reference_radius * 0.5)
        value_channel = hsv[:, :, 2]

        for label_index in range(1, label_count):
            x, y, width, height, area = stats[label_index]
            if area < min_area or area > max_area:
                continue
            if width > max_width or height > max_height:
                continue

            cx, cy = centroids[label_index]
            near_center = float(np.hypot(cx - center_x, cy - center_y)) <= center_distance_limit
            near_border = (
                x <= edge_band
                or y <= edge_band
                or x + width >= frame.shape[1] - edge_band
                or y + height >= frame.shape[0] - edge_band
            )
            if not (near_center or near_border):
                continue

            component_mask = (labels == label_index).astype(np.uint8)
            ring_mask = cv2.dilate(component_mask, np.ones((5, 5), dtype=np.uint8), iterations=1) - component_mask
            ring_pixels = value_channel[ring_mask > 0]
            bright_ring_ratio = float(np.mean(ring_pixels >= 165)) if ring_pixels.size else 0.0
            if not near_center and bright_ring_ratio < 0.12:
                continue

            padding = max(4, int(round(max(width, height) * 0.2)))
            x0 = max(0, x - padding)
            y0 = max(0, y - padding)
            x1 = min(frame.shape[1], x + width + padding)
            y1 = min(frame.shape[0], y + height + padding)
            mask[y0:y1, x0:x1] = 0

    def _build_feature_mask(self, content_mask: np.ndarray) -> np.ndarray:
        feature_mask = content_mask.copy()
        erode_iterations = max(0, int(self.config.minimap_feature_mask_erode))
        if erode_iterations <= 0:
            return feature_mask

        refined = cv2.erode(
            feature_mask,
            np.ones((3, 3), dtype=np.uint8),
            iterations=erode_iterations,
        )
        if np.count_nonzero(refined) < max(64, np.count_nonzero(content_mask) // 2):
            return feature_mask
        return refined

    @staticmethod
    def _resize_mask(mask: np.ndarray | None, ratio: float) -> np.ndarray | None:
        if mask is None:
            return None
        if ratio == 1.0:
            return mask

        height, width = mask.shape[:2]
        resized = cv2.resize(
            mask,
            (max(1, int(width * ratio)), max(1, int(height * ratio))),
            interpolation=cv2.INTER_NEAREST,
        )
        return resized

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
