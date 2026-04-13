from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse
from urllib.request import Request, urlopen

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
    id: str
    title: str
    category_id: int
    longitude: float
    latitude: float
    icon_key: str = ""
    icon_source: str = ""
    category_title: str = ""


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
        icon_dir: str | Path | None = None,
        projection_type: str = "linear",
        tile_zoom: int = 0,
        tile_x_range: tuple[int, int] | None = None,
        tile_y_range: tuple[int, int] | None = None,
        tile_size: int = 256,
        pixel_scale: float = 1.0,
        pixel_offset_x: float = 0.0,
        pixel_offset_y: float = 0.0,
    ) -> None:
        self.pois_path = Path(pois_path)
        self.map_bounds = map_bounds
        self.categories_path = Path(categories_path) if categories_path else None
        self.icon_dir = Path(icon_dir).expanduser() if icon_dir else None
        self.projection_type = projection_type
        self.tile_zoom = tile_zoom
        self.tile_x_range = tile_x_range
        self.tile_y_range = tile_y_range
        self.tile_size = tile_size
        self.pixel_scale = float(pixel_scale)
        self.pixel_offset_x = float(pixel_offset_x)
        self.pixel_offset_y = float(pixel_offset_y)
        self.records = self._load_records(self.pois_path)
        self.categories = self._load_categories(self.categories_path) if self.categories_path else {}
        self._icon_cache: dict[str, np.ndarray | None] = {}
        self._remote_icon_cache_dir = self.pois_path.parent / ".icon_cache"

    def available_categories(self) -> list[tuple[PoiCategory, int]]:
        counts = {}
        for record in self.records:
            counts[record.category_id] = counts.get(record.category_id, 0) + 1

        categories = []
        for category_id, count in counts.items():
            category = self.categories.get(category_id)
            if category is None:
                fallback_title = next(
                    (
                        record.category_title
                        for record in self.records
                        if record.category_id == category_id and record.category_title
                    ),
                    f"分类 {category_id}",
                )
                category = PoiCategory(
                    id=category_id,
                    title=fallback_title,
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
            self._draw_marker(canvas, record, point)

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

        if self.projection_type == "pixel_space":
            x = int(round(longitude * self.pixel_scale + self.pixel_offset_x))
            y = int(round(latitude * self.pixel_scale + self.pixel_offset_y))
            if x < 0 or y < 0 or x >= image_width or y >= image_height:
                return None
            return x, y

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

    def _draw_marker(self, image: np.ndarray, record: PoiRecord, point: tuple[int, int]) -> None:
        icon = self._load_icon(record)
        if icon is not None:
            self._overlay_icon(image, icon, point)
            return

        color = self._category_color(record.category_id)
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

    def _load_icon(self, record: PoiRecord) -> np.ndarray | None:
        category = self.categories.get(record.category_id)
        icon_key = record.icon_key or str(record.category_id)
        icon_value = record.icon_source or (category.icon if category else "")
        cache_key = f"{icon_key}|{icon_value}|{self.icon_dir or ''}"
        if cache_key in self._icon_cache:
            return self._icon_cache[cache_key]

        icon = self._load_icon_from_dir(icon_key)
        if icon is None and icon_value:
            icon = self._load_icon_value(icon_value, icon_key)

        self._icon_cache[cache_key] = icon
        return icon

    def _load_icon_from_dir(self, icon_key: str) -> np.ndarray | None:
        if self.icon_dir is None or not self.icon_dir.exists():
            return None

        candidates = sorted(self.icon_dir.glob(f"{icon_key}.*"))
        candidates.extend(sorted(self.icon_dir.glob(f"{icon_key}_*.*")))
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return self._read_icon(candidate)
        return None

    def _load_icon_value(self, icon_value: str, icon_key: str) -> np.ndarray | None:
        icon_value = str(icon_value).strip()
        if not icon_value:
            return None

        local_candidate = Path(icon_value).expanduser()
        if local_candidate.exists():
            return self._read_icon(local_candidate)

        parsed = urlparse(icon_value)
        if parsed.scheme not in {"http", "https"}:
            return None

        self._remote_icon_cache_dir.mkdir(parents=True, exist_ok=True)
        suffix = Path(parsed.path).suffix or ".png"
        cache_name = f"{icon_key}_{hashlib.sha1(icon_value.encode('utf-8')).hexdigest()[:12]}{suffix}"
        cache_path = self._remote_icon_cache_dir / cache_name
        if not cache_path.exists():
            request = Request(icon_value, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=15) as response:
                cache_path.write_bytes(response.read())
        return self._read_icon(cache_path)

    @staticmethod
    def _read_icon(path: Path) -> np.ndarray | None:
        icon = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
        if icon is None:
            return None
        if icon.ndim == 2:
            icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGRA)
        elif icon.shape[2] == 3:
            alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
            icon = np.dstack([icon, alpha])
        return icon

    @staticmethod
    def _overlay_icon(image: np.ndarray, icon: np.ndarray, point: tuple[int, int]) -> None:
        max_edge = 20
        height, width = icon.shape[:2]
        scale = min(1.0, max_edge / max(height, width, 1))
        if scale != 1.0:
            icon = cv2.resize(
                icon,
                (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
            height, width = icon.shape[:2]

        x0 = int(round(point[0] - width / 2))
        y0 = int(round(point[1] - height / 2))
        x1 = x0 + width
        y1 = y0 + height
        clip_x0 = max(0, x0)
        clip_y0 = max(0, y0)
        clip_x1 = min(image.shape[1], x1)
        clip_y1 = min(image.shape[0], y1)
        if clip_x0 >= clip_x1 or clip_y0 >= clip_y1:
            return

        icon_x0 = clip_x0 - x0
        icon_y0 = clip_y0 - y0
        icon_x1 = icon_x0 + (clip_x1 - clip_x0)
        icon_y1 = icon_y0 + (clip_y1 - clip_y0)

        icon_patch = icon[icon_y0:icon_y1, icon_x0:icon_x1]
        alpha = icon_patch[:, :, 3:4].astype(np.float32) / 255.0
        if np.count_nonzero(alpha) == 0:
            return

        image_patch = image[clip_y0:clip_y1, clip_x0:clip_x1].astype(np.float32)
        blended = alpha * icon_patch[:, :, :3].astype(np.float32) + (1.0 - alpha) * image_patch
        image[clip_y0:clip_y1, clip_x0:clip_x1] = blended.astype(np.uint8)

    @staticmethod
    def _load_records(path: Path) -> list[PoiRecord]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        raw_records = payload["data"] if isinstance(payload, dict) and "data" in payload else payload
        records = []
        for raw in raw_records:
            if "point" in raw and isinstance(raw.get("point"), dict):
                point = raw["point"]
                mark_type = int(raw.get("markType") or raw.get("category_id") or 0)
                mark_title = str(raw.get("markTypeName") or "").strip()
                title = str(raw.get("title") or mark_title or f"类型 {mark_type}")
                records.append(
                    PoiRecord(
                        id=str(raw.get("id") or len(records) + 1),
                        title=title,
                        category_id=mark_type,
                        longitude=float(point.get("lng") or 0.0),
                        latitude=float(point.get("lat") or 0.0),
                        icon_key=str(mark_type) if mark_type else "",
                        icon_source=str(raw.get("iconUrl") or raw.get("icon") or ""),
                        category_title=mark_title,
                    )
                )
                continue
            records.append(
                PoiRecord(
                    id=str(raw.get("id") or len(records) + 1),
                    title=str(raw.get("title") or ""),
                    category_id=int(raw.get("category_id") or 0),
                    longitude=float(raw["longitude"]),
                    latitude=float(raw["latitude"]),
                    icon_key=str(raw.get("category_id") or 0),
                    icon_source="",
                    category_title="",
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
