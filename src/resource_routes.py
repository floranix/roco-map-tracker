from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from statistics import median
from typing import Mapping

import cv2
import numpy as np

from src.collectible_materials import collectible_material_by_id
from src.poi_overlay import PoiCategory, PoiOverlay, PoiRecord


ROUTE_SEGMENT_COLORS = [
    (56, 84, 255),
    (67, 180, 77),
    (255, 154, 46),
    (196, 89, 255),
    (0, 196, 255),
    (255, 114, 196),
]

ROUTE_SELECTION_NONE = "none"


@dataclass
class ResourceRoutePoint:
    id: str
    title: str
    category_id: int
    longitude: float
    latitude: float
    x: float
    y: float
    material_kind: str


@dataclass
class ResourceRouteSegment:
    points: list[ResourceRoutePoint]
    distance: float


@dataclass
class ResourceRoutePlan:
    selection_label: str
    selected_category_ids: list[int]
    source_path: str
    source_label: str
    total_points: int
    total_distance: float
    segments: list[ResourceRouteSegment]
    cached: bool = False

    def text(self) -> str:
        return (
            f"路线: {self.selection_label} | "
            f"{len(self.segments)} 条 | {self.total_points} 点 | "
            f"总长 {self.total_distance:.0f}px | 来源: {self.source_label}"
        )

    def to_dict(self) -> dict:
        return {
            "selection_label": self.selection_label,
            "selected_category_ids": self.selected_category_ids,
            "source_path": self.source_path,
            "source_label": self.source_label,
            "total_points": self.total_points,
            "total_distance": self.total_distance,
            "segments": [
                {
                    "distance": segment.distance,
                    "points": [asdict(point) for point in segment.points],
                }
                for segment in self.segments
            ],
        }

    @staticmethod
    def from_dict(payload: dict) -> ResourceRoutePlan:
        segments = []
        for raw_segment in payload.get("segments", []):
            points = [ResourceRoutePoint(**raw_point) for raw_point in raw_segment.get("points", [])]
            segments.append(
                ResourceRouteSegment(
                    points=points,
                    distance=float(raw_segment.get("distance") or 0.0),
                )
            )
        return ResourceRoutePlan(
            selection_label=str(payload.get("selection_label") or "已选素材"),
            selected_category_ids=[int(value) for value in payload.get("selected_category_ids", [])],
            source_path=str(payload.get("source_path") or ""),
            source_label=str(payload.get("source_label") or "当前点位"),
            total_points=int(payload.get("total_points") or 0),
            total_distance=float(payload.get("total_distance") or 0.0),
            segments=segments,
            cached=True,
        )


def build_resource_route_plan(
    records: list[PoiRecord],
    categories: Mapping[int, PoiCategory],
    overlay: PoiOverlay,
    map_width: int,
    map_height: int,
    selected_category_ids: list[int],
    selection_label: str,
    start_xy: tuple[float, float] | None = None,
    source_label: str = "当前点位",
) -> ResourceRoutePlan:
    normalized_ids = sorted({int(category_id) for category_id in selected_category_ids})
    if not normalized_ids:
        raise ValueError("请先选择需要生成路线的采集素材。")

    points = _collect_route_points(
        records=records,
        categories=categories,
        overlay=overlay,
        map_width=map_width,
        map_height=map_height,
        selected_category_ids=normalized_ids,
    )
    if not points:
        raise ValueError("当前点位数据中没有匹配的采集素材点。")

    segments = _build_route_segments(points, start_xy=start_xy)
    total_distance = sum(segment.distance for segment in segments)
    return ResourceRoutePlan(
        selection_label=selection_label,
        selected_category_ids=normalized_ids,
        source_path=str(overlay.pois_path),
        source_label=source_label,
        total_points=len(points),
        total_distance=total_distance,
        segments=segments,
    )


def render_resource_route_plan(
    image: np.ndarray,
    plan: ResourceRoutePlan | None,
    scale: float = 1.0,
) -> np.ndarray:
    return render_resource_route_plan_viewport(
        image=image,
        plan=plan,
        scale=scale,
        viewport_origin=(0, 0),
    )


def render_resource_route_plan_viewport(
    image: np.ndarray,
    plan: ResourceRoutePlan | None,
    scale: float = 1.0,
    viewport_origin: tuple[int, int] = (0, 0),
) -> np.ndarray:
    if plan is None:
        return image

    canvas = image.copy()
    viewport_x, viewport_y = viewport_origin
    margin = 16
    for segment_index, segment in enumerate(plan.segments):
        if not segment.points:
            continue

        color = ROUTE_SEGMENT_COLORS[segment_index % len(ROUTE_SEGMENT_COLORS)]
        scaled_points = np.array(
            [
                (int(round(point.x * scale)), int(round(point.y * scale)))
                for point in segment.points
            ],
            dtype=np.int32,
        )
        min_x = int(np.min(scaled_points[:, 0]))
        max_x = int(np.max(scaled_points[:, 0]))
        min_y = int(np.min(scaled_points[:, 1]))
        max_y = int(np.max(scaled_points[:, 1]))
        if (
            max_x < viewport_x - margin
            or max_y < viewport_y - margin
            or min_x >= viewport_x + canvas.shape[1] + margin
            or min_y >= viewport_y + canvas.shape[0] + margin
        ):
            continue

        local_points = scaled_points.copy()
        local_points[:, 0] -= viewport_x
        local_points[:, 1] -= viewport_y

        for point_xy in local_points:
            point_tuple = tuple(int(value) for value in point_xy)
            cv2.circle(canvas, point_tuple, 4, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, point_tuple, 3, color, -1, cv2.LINE_AA)

        if len(local_points) >= 2:
            cv2.polylines(
                canvas,
                [local_points.reshape(-1, 1, 2)],
                isClosed=False,
                color=color,
                thickness=3,
                lineType=cv2.LINE_AA,
            )

        start_point = tuple(int(value) for value in local_points[0])
        end_point = tuple(int(value) for value in local_points[-1])
        cv2.circle(canvas, start_point, 9, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(canvas, start_point, 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end_point, 6, (0, 0, 0), -1, cv2.LINE_AA)
        cv2.circle(canvas, end_point, 4, color, -1, cv2.LINE_AA)

    return canvas


def build_route_cache_signature(
    *,
    selected_category_ids: list[int],
    source_path: str,
    source_mtime_ns: int,
    map_path: str,
    map_width: int,
    map_height: int,
    projection_type: str,
    tile_zoom: int,
    tile_x_range: tuple[int, int] | None,
    tile_y_range: tuple[int, int] | None,
    tile_size: int,
    pixel_scale: float,
    pixel_scale_x: float,
    pixel_scale_y: float,
    pixel_offset_x: float,
    pixel_offset_y: float,
) -> dict:
    return {
        "version": 2,
        "selected_category_ids": sorted(int(value) for value in selected_category_ids),
        "source_path": source_path,
        "source_mtime_ns": int(source_mtime_ns),
        "map_path": map_path,
        "map_width": int(map_width),
        "map_height": int(map_height),
        "projection_type": projection_type,
        "tile_zoom": int(tile_zoom),
        "tile_x_range": list(tile_x_range) if tile_x_range is not None else None,
        "tile_y_range": list(tile_y_range) if tile_y_range is not None else None,
        "tile_size": int(tile_size),
        "pixel_scale": float(pixel_scale),
        "pixel_scale_x": float(pixel_scale_x),
        "pixel_scale_y": float(pixel_scale_y),
        "pixel_offset_x": float(pixel_offset_x),
        "pixel_offset_y": float(pixel_offset_y),
    }


def route_cache_path(cache_dir: str | Path, signature: dict) -> Path:
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()
    return cache_root / f"{digest}.json"


def save_route_plan_cache(cache_path: str | Path, signature: dict, plan: ResourceRoutePlan) -> None:
    payload = {
        "signature": signature,
        "plan": plan.to_dict(),
    }
    Path(cache_path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def load_route_plan_cache(cache_path: str | Path, signature: dict) -> ResourceRoutePlan | None:
    path = Path(cache_path)
    if not path.exists():
        return None

    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("signature") != signature:
        return None
    return ResourceRoutePlan.from_dict(payload.get("plan") or {})


def summarize_selection_label(selected_category_ids: list[int]) -> str:
    normalized_ids = sorted({int(value) for value in selected_category_ids})
    if not normalized_ids:
        return "未选素材"

    materials = [collectible_material_by_id(category_id) for category_id in normalized_ids]
    names = [material.name for material in materials if material is not None]
    if not names:
        return f"已选 {len(normalized_ids)} 种素材"
    if len(names) <= 3:
        return "、".join(names)
    return f"已选 {len(names)} 种素材"


def _collect_route_points(
    *,
    records: list[PoiRecord],
    categories: Mapping[int, PoiCategory],
    overlay: PoiOverlay,
    map_width: int,
    map_height: int,
    selected_category_ids: list[int],
) -> list[ResourceRoutePoint]:
    selected = set(selected_category_ids)
    route_points = []
    dedupe_keys = set()

    for record in records:
        if int(record.category_id) not in selected:
            continue

        projected = overlay.project_point(
            longitude=record.longitude,
            latitude=record.latitude,
            image_width=map_width,
            image_height=map_height,
        )
        if projected is None:
            continue

        material = collectible_material_by_id(record.category_id)
        category = categories.get(record.category_id)
        title = str(record.title or (category.title if category is not None else ""))
        if not title and material is not None:
            title = material.name
        if not title:
            title = f"类型 {record.category_id}"

        x, y = projected
        dedupe_key = (record.category_id, int(x), int(y), title.strip())
        if dedupe_key in dedupe_keys:
            continue
        dedupe_keys.add(dedupe_key)

        route_points.append(
            ResourceRoutePoint(
                id=record.id,
                title=title,
                category_id=record.category_id,
                longitude=record.longitude,
                latitude=record.latitude,
                x=float(x),
                y=float(y),
                material_kind=material.kind if material is not None else "",
            )
        )

    return route_points


def _build_route_segments(
    points: list[ResourceRoutePoint],
    start_xy: tuple[float, float] | None,
) -> list[ResourceRouteSegment]:
    if len(points) == 1:
        return [ResourceRouteSegment(points=[points[0]], distance=0.0)]

    clusters = _split_points_into_clusters(points)
    ordered_clusters = _order_clusters(clusters, start_xy=start_xy)

    segments = []
    anchor = start_xy
    for cluster_points in ordered_clusters:
        route_points = _order_points_within_cluster(cluster_points, start_xy=anchor)
        route_distance = _route_distance(route_points)
        segments.append(ResourceRouteSegment(points=route_points, distance=route_distance))
        if route_points:
            anchor = (route_points[-1].x, route_points[-1].y)

    return segments


def _split_points_into_clusters(points: list[ResourceRoutePoint]) -> list[list[ResourceRoutePoint]]:
    if len(points) <= 2:
        return [sorted(points, key=lambda point: (point.y, point.x))]

    edges = _build_mst_edges(points)
    if not edges:
        return [points]

    edge_lengths = [edge[2] for edge in edges]
    threshold = max(180.0, float(median(edge_lengths)) * 2.8)

    parent = list(range(len(points)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parent[root_right] = root_left

    for left, right, distance in edges:
        if distance <= threshold:
            union(left, right)

    groups: dict[int, list[ResourceRoutePoint]] = {}
    for index, point in enumerate(points):
        groups.setdefault(find(index), []).append(point)

    return list(groups.values())


def _build_mst_edges(points: list[ResourceRoutePoint]) -> list[tuple[int, int, float]]:
    count = len(points)
    in_tree = [False] * count
    min_distance_sq = [math.inf] * count
    parent = [-1] * count
    min_distance_sq[0] = 0.0

    coords = np.array([(point.x, point.y) for point in points], dtype=np.float32)
    edges = []

    for _ in range(count):
        current_index = -1
        current_distance_sq = math.inf
        for index in range(count):
            if not in_tree[index] and min_distance_sq[index] < current_distance_sq:
                current_distance_sq = min_distance_sq[index]
                current_index = index

        if current_index < 0:
            break

        in_tree[current_index] = True
        if parent[current_index] >= 0:
            edges.append((parent[current_index], current_index, math.sqrt(current_distance_sq)))

        delta = coords - coords[current_index]
        distances_sq = delta[:, 0] * delta[:, 0] + delta[:, 1] * delta[:, 1]
        for index in range(count):
            if not in_tree[index] and distances_sq[index] < min_distance_sq[index]:
                min_distance_sq[index] = float(distances_sq[index])
                parent[index] = current_index

    return edges


def _order_clusters(
    clusters: list[list[ResourceRoutePoint]],
    start_xy: tuple[float, float] | None,
) -> list[list[ResourceRoutePoint]]:
    def cluster_key(points: list[ResourceRoutePoint]) -> tuple[float, float]:
        centroid_x = sum(point.x for point in points) / max(len(points), 1)
        centroid_y = sum(point.y for point in points) / max(len(points), 1)
        if start_xy is None:
            return (-len(points), centroid_y + centroid_x)
        return (
            _distance_sq((centroid_x, centroid_y), start_xy),
            -len(points),
        )

    return sorted(clusters, key=cluster_key)


def _order_points_within_cluster(
    points: list[ResourceRoutePoint],
    start_xy: tuple[float, float] | None,
) -> list[ResourceRoutePoint]:
    remaining = list(points)
    if not remaining:
        return []

    if start_xy is None:
        current = min(remaining, key=lambda point: (point.y, point.x))
    else:
        current = min(remaining, key=lambda point: _distance_sq((point.x, point.y), start_xy))
    route = [current]
    remaining.remove(current)

    while remaining:
        current = min(remaining, key=lambda point: _distance_sq((point.x, point.y), (route[-1].x, route[-1].y)))
        route.append(current)
        remaining.remove(current)

    if len(route) <= 80:
        route = _improve_route_with_2opt(route)
    return route


def _improve_route_with_2opt(points: list[ResourceRoutePoint]) -> list[ResourceRoutePoint]:
    route = list(points)
    if len(route) < 4:
        return route

    improved = True
    while improved:
        improved = False
        for left in range(1, len(route) - 2):
            for right in range(left + 1, len(route) - 1):
                before = _distance(route[left - 1], route[left]) + _distance(route[right], route[right + 1])
                after = _distance(route[left - 1], route[right]) + _distance(route[left], route[right + 1])
                if after + 1e-6 < before:
                    route[left : right + 1] = reversed(route[left : right + 1])
                    improved = True
        if len(route) > 40:
            break
    return route


def _route_distance(points: list[ResourceRoutePoint]) -> float:
    if len(points) < 2:
        return 0.0
    return sum(_distance(points[index - 1], points[index]) for index in range(1, len(points)))


def _distance(left: ResourceRoutePoint, right: ResourceRoutePoint) -> float:
    return math.sqrt(_distance_sq((left.x, left.y), (right.x, right.y)))


def _distance_sq(left: tuple[float, float], right: tuple[float, float]) -> float:
    dx = left[0] - right[0]
    dy = left[1] - right[1]
    return dx * dx + dy * dy
