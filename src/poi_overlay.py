from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


DEFAULT_CATEGORY_COLOR = (60, 220, 255)


@dataclass
class PoiCategory:
    id: int
    title: str
    group_id: int
    icon: str = ""
    group_title: str = ""


@dataclass
class PoiRecord:
    id: int
    title: str
    category_id: int
    longitude: float
    latitude: float


@dataclass
class PoiRenderOptions:
    enabled: bool = False
    selected_category_ids: Optional[list[int]] = None
    keyword: str = ""
    show_labels: bool = False
    max_points: int = 1500
    label_limit: int = 40


@dataclass
class PoiRenderSummary:
    total: int
    matched: int
    rendered: int
    note: str = ""

    def text(self) -> str:
        summary = f"点位: {self.rendered}/{self.matched}，总计 {self.total}"
        if self.note:
            return f"{summary}（{self.note}）"
        return summary


class PoiOverlay:
    def __init__(
        self,
        pois_path: str | Path,
        map_bounds: tuple[float, float, float, float],
        categories_path: str | Path | None = None,
        projection_type: str = "linear",
        tile_zoom: int = 0,
        tile_x_range: tuple[int, int] | None = None,
        tile_y_range: tuple[int, int] | None = None,
        tile_size: int = 256,
    ) -> None:
        self.pois_path = Path(pois_path)
        self.map_bounds = map_bounds
        self.categories_path = Path(categories_path) if categories_path else None
        self.projection_type = projection_type
        self.tile_zoom = tile_zoom
        self.tile_x_range = tile_x_range
        self.tile_y_range = tile_y_range
        self.tile_size = tile_size
        self.records = self._load_records(self.pois_path)
        self.categories = self._load_categories(self.categories_path) if self.categories_path else {}

    def available_categories(self) -> list[tuple[PoiCategory, int]]:
        counts = {}
        for record in self.records:
            counts[record.category_id] = counts.get(record.category_id, 0) + 1

        categories = []
        for category_id, count in counts.items():
            category = self.categories.get(category_id)
            if category is None:
                category = PoiCategory(
                    id=category_id,
                    title=f"分类 {category_id}",
                    group_id=0,
                    group_title="未分组",
                )
            categories.append((category, count))

        categories.sort(key=lambda item: (item[0].group_title, item[0].title, item[0].id))
        return categories

    def render_map(
        self,
        map_image: np.ndarray,
        options: PoiRenderOptions,
        focus_xy: tuple[float, float] | None = None,
    ) -> tuple[np.ndarray, PoiRenderSummary | None]:
        if not options.enabled:
            return map_image, None

        selected_category_ids = options.selected_category_ids or []
        if not selected_category_ids:
            return map_image, PoiRenderSummary(
                total=len(self.records),
                matched=0,
                rendered=0,
                note="未选分类",
            )

        matched = self._filter_records(
            selected_category_ids=selected_category_ids,
            keyword=options.keyword,
        )
        if not matched:
            return map_image, PoiRenderSummary(
                total=len(self.records),
                matched=0,
                rendered=0,
                note="无匹配",
            )

        rendered_records = matched[: options.max_points]
        canvas = map_image.copy()
        points = []
        for record in rendered_records:
            point = self.project_point(
                longitude=record.longitude,
                latitude=record.latitude,
                image_width=canvas.shape[1],
                image_height=canvas.shape[0],
            )
            if point is None:
                continue
            points.append((record, point))
            self._draw_marker(canvas, record.category_id, point)

        if options.show_labels and points:
            for record, point in self._pick_label_records(points, focus_xy, options.label_limit):
                self._draw_label(canvas, record.title, point, record.category_id)

        summary = PoiRenderSummary(
            total=len(self.records),
            matched=len(matched),
            rendered=len(points),
        )
        return canvas, summary

    def project_point(
        self,
        longitude: float,
        latitude: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int] | None:
        if image_width <= 0 or image_height <= 0:
            return None

        if self.projection_type == "web_mercator_tiles":
            return self._project_web_mercator_point(longitude, latitude, image_width, image_height)

        min_lon, min_lat, max_lon, max_lat = self.map_bounds
        if max_lon <= min_lon or max_lat <= min_lat:
            return None

        x_ratio = (longitude - min_lon) / (max_lon - min_lon)
        y_ratio = (max_lat - latitude) / (max_lat - min_lat)
        x = int(round(x_ratio * max(0, image_width - 1)))
        y = int(round(y_ratio * max(0, image_height - 1)))
        if x < 0 or y < 0 or x >= image_width or y >= image_height:
            return None
        return x, y

    def _project_web_mercator_point(
        self,
        longitude: float,
        latitude: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int] | None:
        if (
            self.tile_zoom <= 0
            or self.tile_x_range is None
            or self.tile_y_range is None
            or len(self.tile_x_range) != 2
            or len(self.tile_y_range) != 2
        ):
            return None

        x_min, x_max = self.tile_x_range
        y_min, y_max = self.tile_y_range
        if x_max < x_min or y_max < y_min:
            return None

        x_pixel = (self._lon_to_tile_x(longitude, self.tile_zoom) - x_min) * self.tile_size
        y_pixel = (self._lat_to_tile_y(latitude, self.tile_zoom) - y_min) * self.tile_size
        return self._clamp_pixel_point(x_pixel, y_pixel, image_width, image_height)

    @staticmethod
    def _clamp_pixel_point(
        x_pixel: float,
        y_pixel: float,
        image_width: int,
        image_height: int,
    ) -> tuple[int, int] | None:
        if image_width <= 0 or image_height <= 0:
            return None

        x = int(round(x_pixel))
        y = int(round(y_pixel))
        x = min(max(x, 0), image_width - 1)
        y = min(max(y, 0), image_height - 1)
        return x, y

    @staticmethod
    def _lon_to_tile_x(longitude: float, zoom: int) -> float:
        return (longitude + 180.0) / 360.0 * (2**zoom)

    @staticmethod
    def _lat_to_tile_y(latitude: float, zoom: int) -> float:
        latitude = max(min(latitude, 85.05112878), -85.05112878)
        latitude_rad = math.radians(latitude)
        return (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * (2**zoom)

    def _filter_records(self, selected_category_ids: list[int], keyword: str) -> list[PoiRecord]:
        keyword = keyword.strip().lower()
        selected = set(selected_category_ids)

        def matches(record: PoiRecord) -> bool:
            if selected and record.category_id not in selected:
                return False
            if keyword and keyword not in record.title.lower():
                return False
            return True

        return [record for record in self.records if matches(record)]

    def _pick_label_records(
        self,
        points: list[tuple[PoiRecord, tuple[int, int]]],
        focus_xy: tuple[float, float] | None,
        label_limit: int,
    ) -> list[tuple[PoiRecord, tuple[int, int]]]:
        if label_limit <= 0:
            return []

        if focus_xy is None:
            return points[:label_limit]

        focus_x, focus_y = focus_xy
        ranked = sorted(
            points,
            key=lambda item: (item[1][0] - focus_x) ** 2 + (item[1][1] - focus_y) ** 2,
        )
        return ranked[:label_limit]

    def _draw_marker(self, image: np.ndarray, category_id: int, point: tuple[int, int]) -> None:
        color = self._category_color(category_id)
        cv2.circle(image, point, 5, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(image, point, 4, color, -1, cv2.LINE_AA)

    def _draw_label(
        self,
        image: np.ndarray,
        text: str,
        point: tuple[int, int],
        category_id: int,
    ) -> None:
        color = self._category_color(category_id)
        label_point = (point[0] + 8, point[1] - 8)
        cv2.putText(
            image,
            text,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            label_point,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def _category_color(self, category_id: int) -> tuple[int, int, int]:
        if category_id <= 0:
            return DEFAULT_CATEGORY_COLOR
        return (
            50 + (category_id * 37) % 180,
            50 + (category_id * 67) % 180,
            50 + (category_id * 97) % 180,
        )

    @staticmethod
    def _load_records(path: Path) -> list[PoiRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        records = []
        for raw in raw_records:
            records.append(
                PoiRecord(
                    id=int(raw["id"]),
                    title=str(raw.get("title") or ""),
                    category_id=int(raw.get("category_id") or 0),
                    longitude=float(raw["longitude"]),
                    latitude=float(raw["latitude"]),
                )
            )
        return records

    @staticmethod
    def _load_categories(path: Path) -> dict[int, PoiCategory]:
        groups = json.loads(path.read_text(encoding="utf-8"))
        categories: dict[int, PoiCategory] = {}
        for group in groups:
            group_title = str(group.get("title") or "")
            for raw_category in group.get("categories", []):
                category = PoiCategory(
                    id=int(raw_category["id"]),
                    title=str(raw_category.get("title") or ""),
                    group_id=int(raw_category.get("group_id") or group.get("id") or 0),
                    icon=str(raw_category.get("icon") or ""),
                    group_title=group_title,
                )
                categories[category.id] = category
        return categories
