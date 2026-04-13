from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import math

import cv2
import numpy as np

from src.utils import LocalizationResult


ROCOM_17173_SPACE = "rocom_17173"
ROCOM_BILIWIKI_SPACE = "rocom_biliwiki"

MAP_COORDINATE_SPACES = {
    "rocom-shijie-z13.png": ROCOM_17173_SPACE,
    "rocom_base_z8.png": ROCOM_BILIWIKI_SPACE,
    "rocom_caiji_overlay.png": ROCOM_BILIWIKI_SPACE,
    "rocom_poi_overlay.png": ROCOM_BILIWIKI_SPACE,
}

# 17173 逻辑底图 -> biliwiki 底图/overlay 的稳定配准矩阵。
# 该矩阵基于两张完整地图的全图特征匹配与 RANSAC 估计得到。
ROCOM_17173_TO_BILIWIKI_H = np.array(
    [
        [1.99720688, -0.00082095934, -1940.24364],
        [-0.000128190243, 1.99554078, -4435.49101],
        [-9.6557274e-08, -1.98053379e-07, 1.0],
    ],
    dtype=np.float64,
)


class MapAlignment:
    def __init__(self, source_space: str, target_space: str, matrix: np.ndarray) -> None:
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError("地图对齐矩阵必须为 3x3。")

        self.source_space = source_space
        self.target_space = target_space
        self.matrix = matrix
        self.inverse_matrix = np.linalg.inv(matrix)

    def project_point(self, x: float, y: float) -> tuple[float, float] | None:
        if not _is_finite_number(x) or not _is_finite_number(y):
            return None

        projected = cv2.perspectiveTransform(
            np.array([[[float(x), float(y)]]], dtype=np.float32),
            self.matrix.astype(np.float32),
        ).reshape(-1, 2)[0]
        return float(projected[0]), float(projected[1])

    def project_points(
        self,
        points: list[tuple[float, float]] | tuple[tuple[float, float], ...],
    ) -> list[tuple[float, float]]:
        if not points:
            return []

        array = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(array, self.matrix.astype(np.float32)).reshape(-1, 2)
        return [(float(x), float(y)) for x, y in projected]

    def project_result(self, result: LocalizationResult | None) -> LocalizationResult | None:
        if result is None:
            return None

        projected_center = self.project_point(result.x, result.y)
        projected_corners = None
        if result.corners:
            projected_corners = self.project_points(result.corners)

        projected_bbox = None
        if projected_corners:
            projected_bbox = _bbox_from_points(projected_corners)
        elif result.bbox is not None:
            x0, y0, x1, y1 = result.bbox
            projected_bbox_corners = self.project_points(
                [
                    (float(x0), float(y0)),
                    (float(x1), float(y0)),
                    (float(x1), float(y1)),
                    (float(x0), float(y1)),
                ]
            )
            projected_bbox = _bbox_from_points(projected_bbox_corners)

        x = math.nan
        y = math.nan
        if projected_center is not None:
            x, y = projected_center

        return replace(
            result,
            x=x,
            y=y,
            bbox=projected_bbox,
            corners=projected_corners,
        )


def resolve_map_alignment(source_map_path: str, target_map_path: str) -> MapAlignment | None:
    source_space = _map_coordinate_space(source_map_path)
    target_space = _map_coordinate_space(target_map_path)
    if not source_space or not target_space:
        return None

    if source_space == target_space:
        return MapAlignment(source_space, target_space, np.eye(3, dtype=np.float64))

    if source_space == ROCOM_17173_SPACE and target_space == ROCOM_BILIWIKI_SPACE:
        return MapAlignment(source_space, target_space, ROCOM_17173_TO_BILIWIKI_H)

    if source_space == ROCOM_BILIWIKI_SPACE and target_space == ROCOM_17173_SPACE:
        return MapAlignment(source_space, target_space, np.linalg.inv(ROCOM_17173_TO_BILIWIKI_H))

    return None


def _map_coordinate_space(map_path: str | Path) -> str:
    filename = Path(map_path).expanduser().name.lower()
    return MAP_COORDINATE_SPACES.get(filename, "")


def _bbox_from_points(points: list[tuple[float, float]]) -> tuple[int, int, int, int] | None:
    if not points:
        return None

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]
    return (
        int(round(min(xs))),
        int(round(min(ys))),
        int(round(max(xs))),
        int(round(max(ys))),
    )


def _is_finite_number(value: float | None) -> bool:
    return value is not None and math.isfinite(value) and not math.isnan(value)
