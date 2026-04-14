from __future__ import annotations

from dataclasses import dataclass
import math

import cv2
import numpy as np


@dataclass(frozen=True)
class MapPyramidLevel:
    scale: float
    image: np.ndarray


class MapPyramid:
    def __init__(
        self,
        source_image: np.ndarray,
        min_long_edge: int = 1024,
        max_levels: int = 6,
    ) -> None:
        if source_image.ndim not in (2, 3):
            raise ValueError("地图图像维度不合法。")

        self.source_image = source_image
        self.levels = self._build_levels(
            source_image=source_image,
            min_long_edge=max(128, int(min_long_edge)),
            max_levels=max(1, int(max_levels)),
        )

    @staticmethod
    def _build_levels(
        source_image: np.ndarray,
        min_long_edge: int,
        max_levels: int,
    ) -> list[MapPyramidLevel]:
        levels = [MapPyramidLevel(scale=1.0, image=source_image)]
        current_image = source_image
        current_scale = 1.0

        while len(levels) < max_levels:
            height, width = current_image.shape[:2]
            if max(height, width) <= min_long_edge:
                break

            next_width = max(1, width // 2)
            next_height = max(1, height // 2)
            next_image = cv2.resize(
                current_image,
                (next_width, next_height),
                interpolation=cv2.INTER_AREA,
            )
            current_scale *= 0.5
            levels.append(MapPyramidLevel(scale=current_scale, image=next_image))
            current_image = next_image

        return levels

    def select_level(self, target_scale: float) -> MapPyramidLevel:
        normalized_scale = max(float(target_scale), 1e-6)
        for level in reversed(self.levels):
            if level.scale >= normalized_scale:
                return level
        return self.levels[0]

    def render_viewport(
        self,
        target_scale: float,
        view_origin: tuple[int, int],
        canvas_width: int,
        canvas_height: int,
    ) -> tuple[np.ndarray, MapPyramidLevel]:
        level = self.select_level(target_scale)
        level_scale = max(level.scale, 1e-6)
        relative_scale = max(float(target_scale) / level_scale, 1e-6)
        view_left, view_top = view_origin

        source_image = level.image
        source_height, source_width = source_image.shape[:2]

        src_x0 = max(0, min(source_width - 1, int(math.floor(view_left / relative_scale))))
        src_y0 = max(0, min(source_height - 1, int(math.floor(view_top / relative_scale))))
        src_x1 = min(
            source_width,
            max(src_x0 + 1, int(math.ceil((view_left + canvas_width) / relative_scale)) + 1),
        )
        src_y1 = min(
            source_height,
            max(src_y0 + 1, int(math.ceil((view_top + canvas_height) / relative_scale)) + 1),
        )

        crop = source_image[src_y0:src_y1, src_x0:src_x1]
        scaled_crop_width = max(1, int(math.ceil((src_x1 - src_x0) * relative_scale)))
        scaled_crop_height = max(1, int(math.ceil((src_y1 - src_y0) * relative_scale)))
        interpolation = cv2.INTER_AREA if relative_scale < 1.0 else cv2.INTER_LINEAR
        scaled_crop = cv2.resize(crop, (scaled_crop_width, scaled_crop_height), interpolation=interpolation)

        crop_origin_x = int(round(src_x0 * relative_scale))
        crop_origin_y = int(round(src_y0 * relative_scale))
        offset_x = max(0, view_left - crop_origin_x)
        offset_y = max(0, view_top - crop_origin_y)
        viewport = scaled_crop[offset_y : offset_y + canvas_height, offset_x : offset_x + canvas_width]

        if viewport.shape[:2] == (canvas_height, canvas_width):
            return viewport, level

        if source_image.ndim == 2:
            padded = np.zeros((canvas_height, canvas_width), dtype=source_image.dtype)
        else:
            padded = np.zeros((canvas_height, canvas_width, source_image.shape[2]), dtype=source_image.dtype)
        padded[: viewport.shape[0], : viewport.shape[1]] = viewport
        return padded, level
