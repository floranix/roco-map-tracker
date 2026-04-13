from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field, fields
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import yaml


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
POSITION_MARKER_ICON_PATH = Path(
    "/Users/lanier/IdeaProjects/violet-jb-2/composeApp/src/commonMain/composeResources/drawable/ic_roco_def_head.png"
)
STATE_LABELS = {
    "tracking": "跟踪中",
    "relocalizing": "重定位中",
    "lost": "已丢失",
}
METHOD_LABELS = {
    "global_feature_match": "全局特征匹配",
    "global_tile_match": "全局分块匹配",
    "global_template_match": "全局模板匹配",
    "local_feature_match": "局部特征匹配",
    "local_template_match": "局部模板匹配",
    "prediction": "运动预测",
}


@dataclass
class AppConfig:
    map_path: str = "data/full_map.png"
    display_map_path: str = ""
    map_projection: str = "linear"
    map_tile_zoom: int = 0
    map_tile_x_range: list[int] = field(default_factory=list)
    map_tile_y_range: list[int] = field(default_factory=list)
    map_tile_size: int = 256
    frame_preprocess_mode: str = "none"
    minimap_outer_margin: int = 10
    minimap_center_mask_ratio: float = 0.12
    minimap_icon_mask_ratio: float = 0.18
    minimap_icon_offset_x_ratio: float = -0.65
    minimap_icon_offset_y_ratio: float = 0.55
    minimap_feature_mask_erode: int = 2
    global_search_scales: list[float] = field(
        default_factory=lambda: [1.0, 0.85, 0.7, 0.55, 0.45, 0.35, 0.25, 1.2, 1.4, 1.7, 2.0]
    )
    global_tile_size: int = 1024
    global_tile_stride: int = 768
    global_tile_top_k: int = 8
    feature_type: str = "orb"
    min_match_count: int = 12
    roi_expand_pixels: int = 120
    tracking_template_scales: list[float] = field(default_factory=lambda: [0.97, 1.0, 1.03])
    tracking_template_top_per_scale: int = 1
    tracking_template_top_k: int = 2
    tracking_template_refine_radius: int = 180
    tracking_template_min_score: float = 0.76
    tracking_template_early_accept_score: float = 0.80
    tracking_motion_gate_pixels: float = 210.0
    tracking_motion_gate_per_lost_frame: float = 45.0
    resize_ratio: float = 1.0
    use_optical_flow: bool = True
    use_kalman: bool = True
    max_lost_frames: int = 10
    ratio_test: float = 0.75
    ransac_threshold: float = 5.0
    orb_nfeatures: int = 5000
    sift_nfeatures: int = 1500
    max_rotation_degrees: float = 8.0
    candidate_verification_weight: float = 0.55
    use_template_matching: bool = True
    template_match_map_downsample: float = 0.25
    template_match_scales: list[float] = field(
        default_factory=lambda: [0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.1, 1.2, 1.35, 1.5]
    )
    template_match_top_per_scale: int = 2
    template_match_top_k: int = 6
    template_match_refine_radius: int = 420
    template_match_min_score: float = 0.72
    template_match_blur_size: int = 7
    output_dir: str = "outputs"
    save_visualizations: bool = False
    map_bounds: list[float] = field(default_factory=list)
    poi_data_path: str = ""
    poi_categories_path: str = ""
    poi_icon_dir: str = ""
    poi_pixel_scale: float = 1.0
    poi_pixel_scale_x: float = 0.0
    poi_pixel_scale_y: float = 0.0
    poi_pixel_offset_x: float = 0.0
    poi_pixel_offset_y: float = 0.0
    show_poi_overlay: bool = False
    show_poi_labels: bool = False
    poi_keyword: str = ""
    poi_category_ids: list[int] = field(default_factory=list)
    poi_max_draw: int = 1500
    poi_label_limit: int = 40
    capture_interval_ms: int = 250
    capture_region: list[int] = field(default_factory=list)


@dataclass
class LocalizationResult:
    x: float
    y: float
    theta: Optional[float]
    score: float
    state: str
    method: str
    matches: int = 0
    inliers: int = 0
    bbox: Optional[tuple[int, int, int, int]] = None
    corners: Optional[list[tuple[float, float]]] = None

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["x"] = _rounded_or_none(self.x)
        payload["y"] = _rounded_or_none(self.y)
        payload["theta"] = _rounded_or_none(self.theta)
        payload["score"] = _rounded_or_none(self.score)
        payload["state_text"] = localize_state(self.state)
        payload["method_text"] = localize_method(self.method)
        return payload


def _rounded_or_none(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    return round(float(value), 3)


def load_config(path: str | Path) -> AppConfig:
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"配置文件不存在：{config_path}")

    raw_data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    allowed_fields = {field.name for field in fields(AppConfig)}
    filtered = {key: value for key, value in raw_data.items() if key in allowed_fields}
    return AppConfig(**filtered)


def guess_poi_categories_path(
    poi_data_path: str | Path,
    explicit_path: str | Path | None = None,
) -> str:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists():
            return str(candidate)

    if not str(poi_data_path or "").strip():
        return ""

    poi_path = Path(poi_data_path).expanduser()
    if not poi_path.exists():
        return ""

    sibling = poi_path.with_name("categories.json")
    return str(sibling) if sibling.exists() else ""


def guess_poi_icon_dir(
    poi_data_path: str | Path,
    explicit_path: str | Path | None = None,
) -> str:
    if explicit_path:
        candidate = Path(explicit_path).expanduser()
        if candidate.exists() and candidate.is_dir():
            return str(candidate)

    if not str(poi_data_path or "").strip():
        return ""

    poi_path = Path(poi_data_path).expanduser()
    if not poi_path.exists():
        return ""

    for sibling_name in ("icons", "icon_cache", "icon_cache_caiji", "icon_cache_wiki"):
        sibling = poi_path.with_name(sibling_name)
        if sibling.exists() and sibling.is_dir():
            return str(sibling)
    return ""


def load_map_bounds_from_metadata(poi_data_path: str | Path) -> list[float]:
    metadata = load_map_metadata_from_poi_data(poi_data_path)
    bounds = metadata.get("bounds")
    if not isinstance(bounds, list) or len(bounds) != 4:
        return []

    try:
        return [float(value) for value in bounds]
    except (TypeError, ValueError):
        return []


def load_map_metadata_from_poi_data(poi_data_path: str | Path) -> dict[str, Any]:
    poi_path = Path(poi_data_path).expanduser()
    if not poi_path.exists():
        return {}

    metadata_path = poi_path.with_name("metadata.json")
    if not metadata_path.exists():
        return {}

    try:
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def apply_map_metadata_defaults(config: AppConfig) -> AppConfig:
    if not config.poi_data_path:
        return config

    metadata = load_map_metadata_from_poi_data(config.poi_data_path)
    if not metadata:
        return config

    if (not config.map_bounds or len(config.map_bounds) != 4) and isinstance(metadata.get("bounds"), list):
        config.map_bounds = [float(value) for value in metadata["bounds"]]

    if config.map_projection == "linear" and metadata.get("stitched_map_projection"):
        config.map_projection = str(metadata["stitched_map_projection"])

    if not config.poi_icon_dir and metadata.get("poi_icon_dir"):
        icon_dir = Path(config.poi_data_path).expanduser().with_name(str(metadata["poi_icon_dir"]))
        if icon_dir.exists() and icon_dir.is_dir():
            config.poi_icon_dir = str(icon_dir)

    if not config.poi_icon_dir:
        config.poi_icon_dir = guess_poi_icon_dir(config.poi_data_path, config.poi_icon_dir)

    if metadata.get("poi_pixel_scale") is not None and config.poi_pixel_scale == 1.0:
        config.poi_pixel_scale = float(metadata["poi_pixel_scale"])

    if metadata.get("poi_pixel_offset_x") is not None and config.poi_pixel_offset_x == 0.0:
        config.poi_pixel_offset_x = float(metadata["poi_pixel_offset_x"])

    if metadata.get("poi_pixel_offset_y") is not None and config.poi_pixel_offset_y == 0.0:
        config.poi_pixel_offset_y = float(metadata["poi_pixel_offset_y"])

    tile_grid = metadata.get("max_zoom_tile_grid")
    if isinstance(tile_grid, dict):
        if not config.map_tile_zoom:
            config.map_tile_zoom = int(tile_grid.get("zoom") or 0)
        if not config.map_tile_x_range:
            config.map_tile_x_range = [int(tile_grid.get("x_min")), int(tile_grid.get("x_max"))]
        if not config.map_tile_y_range:
            config.map_tile_y_range = [int(tile_grid.get("y_min")), int(tile_grid.get("y_max"))]
        if not config.map_tile_size:
            config.map_tile_size = int(tile_grid.get("tile_size") or 256)

    stitched_map_file = metadata.get("stitched_map_file")
    if stitched_map_file:
        stitched_map_path = Path(config.poi_data_path).expanduser().with_name(str(stitched_map_file))
        if stitched_map_path.exists() and (not config.map_path or Path(config.map_path) == Path("data/full_map.png")):
            config.map_path = str(stitched_map_path)

    return config


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def load_image(path: str | Path, grayscale: bool = False) -> np.ndarray:
    image_path = Path(path)
    if not image_path.exists():
        raise FileNotFoundError(f"图像文件不存在：{image_path}")

    mode = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
    image = cv2.imread(str(image_path), mode)
    if image is None:
        raise ValueError(f"读取图像失败：{image_path}")
    return image


@lru_cache(maxsize=4)
def load_position_marker_icon(path: str | Path = POSITION_MARKER_ICON_PATH) -> np.ndarray | None:
    image_path = Path(path).expanduser()
    if not image_path.exists():
        return None

    icon = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if icon is None:
        return None
    if icon.ndim == 2:
        icon = cv2.cvtColor(icon, cv2.COLOR_GRAY2BGRA)
    elif icon.shape[2] == 3:
        alpha = np.full(icon.shape[:2], 255, dtype=np.uint8)
        icon = np.dstack([icon, alpha])
    return icon


def resize_image(image: np.ndarray, ratio: float) -> np.ndarray:
    if ratio == 1.0:
        return image

    height, width = image.shape[:2]
    resized = cv2.resize(
        image,
        (max(1, int(width * ratio)), max(1, int(height * ratio))),
        interpolation=cv2.INTER_AREA if ratio < 1.0 else cv2.INTER_LINEAR,
    )
    return resized


def list_image_files(directory: str | Path) -> list[Path]:
    directory_path = Path(directory)
    if not directory_path.exists():
        raise FileNotFoundError(f"输入目录不存在：{directory_path}")

    files = [path for path in sorted(directory_path.iterdir()) if path.suffix.lower() in IMAGE_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"输入目录中没有可用图像：{directory_path}")
    return files


def result_json(frame_name: str, result: LocalizationResult) -> str:
    payload = {"frame": frame_name, **result.to_dict()}
    return json.dumps(payload, ensure_ascii=False)


def format_result_text(frame_name: str, result: LocalizationResult) -> str:
    x = _display_number(result.x)
    y = _display_number(result.y)
    theta = _display_number(result.theta)
    return (
        f"帧: {frame_name} | 状态: {localize_state(result.state)} | "
        f"位置: ({x}, {y}) | 角度: {theta} | "
        f"置信度: {result.score:.3f} | 方法: {localize_method(result.method)}"
    )


def draw_localization(
    map_image: np.ndarray,
    frame_image: np.ndarray,
    result: LocalizationResult,
    extra_lines: Optional[list[str]] = None,
    max_panel_height: int = 720,
) -> np.ndarray:
    annotated_map = map_image.copy()

    if not math.isnan(result.x) and not math.isnan(result.y):
        center = (int(round(result.x)), int(round(result.y)))
        draw_position_marker(annotated_map, center, target_edge=34)

    annotated_frame = frame_image.copy()

    focus_map, overview_map = _build_focus_map_view(annotated_map, frame_image, result)
    if overview_map is not None:
        _overlay_overview_map(focus_map, overview_map)

    annotated_map = _fit_to_height(focus_map, max_panel_height)
    annotated_frame = _fit_to_height(annotated_frame, annotated_map.shape[0])

    frame_height = annotated_frame.shape[0]
    map_height = annotated_map.shape[0]
    canvas_height = max(map_height, frame_height)
    canvas_width = annotated_map.shape[1] + annotated_frame.shape[1]

    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)
    canvas[:map_height, : annotated_map.shape[1]] = annotated_map
    canvas[:frame_height, annotated_map.shape[1] :] = annotated_frame
    return canvas


def offset_result_geometry(result: LocalizationResult, dx: float, dy: float) -> LocalizationResult:
    if result.bbox is not None:
        x0, y0, x1, y1 = result.bbox
        result.bbox = (
            int(round(x0 + dx)),
            int(round(y0 + dy)),
            int(round(x1 + dx)),
            int(round(y1 + dy)),
        )

    if result.corners is not None:
        result.corners = [(x + dx, y + dy) for x, y in result.corners]
    return result


def localize_state(state: str) -> str:
    return STATE_LABELS.get(state, state)


def localize_method(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def _fit_to_height(image: np.ndarray, target_height: int) -> np.ndarray:
    height, width = image.shape[:2]
    if height <= target_height:
        return image
    ratio = target_height / height
    return cv2.resize(image, (max(1, int(width * ratio)), target_height), interpolation=cv2.INTER_AREA)


def _display_number(value: Optional[float]) -> str:
    rounded = _rounded_or_none(value)
    return "未知" if rounded is None else str(rounded)


def _build_focus_map_view(
    annotated_map: np.ndarray,
    frame_image: np.ndarray,
    result: LocalizationResult,
) -> tuple[np.ndarray, np.ndarray | None]:
    map_height, map_width = annotated_map.shape[:2]
    frame_height, frame_width = frame_image.shape[:2]
    center_x = int(round(result.x)) if _is_finite_number(result.x) else map_width // 2
    center_y = int(round(result.y)) if _is_finite_number(result.y) else map_height // 2

    bbox_width, bbox_height = _result_extent(result, frame_width, frame_height)
    crop_width = min(map_width, max(frame_width * 2, int(bbox_width * 3), 720))
    crop_height = min(map_height, max(frame_height * 2, int(bbox_height * 3), 720))

    x0 = min(max(0, center_x - crop_width // 2), max(0, map_width - crop_width))
    y0 = min(max(0, center_y - crop_height // 2), max(0, map_height - crop_height))
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    focus_map = annotated_map[y0:y1, x0:x1].copy()
    if crop_width >= map_width and crop_height >= map_height:
        return focus_map, None

    overview_map = _build_overview_map(
        annotated_map=annotated_map,
        focus_rect=(x0, y0, x1, y1),
        result=result,
    )
    return focus_map, overview_map


def _result_extent(
    result: LocalizationResult,
    fallback_width: int,
    fallback_height: int,
) -> tuple[int, int]:
    if result.bbox is not None:
        x0, y0, x1, y1 = result.bbox
        return max(1, x1 - x0), max(1, y1 - y0)

    if result.corners:
        corners = np.array(result.corners, dtype=np.float32)
        x_values = corners[:, 0]
        y_values = corners[:, 1]
        return (
            max(1, int(np.ceil(x_values.max() - x_values.min()))),
            max(1, int(np.ceil(y_values.max() - y_values.min()))),
        )

    return fallback_width, fallback_height


def _build_overview_map(
    annotated_map: np.ndarray,
    focus_rect: tuple[int, int, int, int],
    result: LocalizationResult,
) -> np.ndarray:
    map_height, map_width = annotated_map.shape[:2]
    preview_max = 220
    scale = min(preview_max / map_width, preview_max / map_height)
    scale = min(scale, 1.0)
    preview_width = max(1, int(round(map_width * scale)))
    preview_height = max(1, int(round(map_height * scale)))
    overview = cv2.resize(annotated_map, (preview_width, preview_height), interpolation=cv2.INTER_AREA)

    if _is_finite_number(result.x) and _is_finite_number(result.y):
        point = (int(round(result.x * scale)), int(round(result.y * scale)))
        draw_position_marker(overview, point, target_edge=14)

    return overview


def _overlay_overview_map(image: np.ndarray, overview_map: np.ndarray) -> None:
    image_height, image_width = image.shape[:2]
    inset_height, inset_width = overview_map.shape[:2]
    margin = 16
    padding = 8

    panel_width = inset_width + padding * 2
    panel_height = inset_height + padding * 2
    x0 = max(margin, image_width - panel_width - margin)
    y0 = margin
    x1 = min(image_width, x0 + panel_width)
    y1 = min(image_height, y0 + panel_height)

    panel = image[y0:y1, x0:x1]
    panel[:] = np.clip(panel * 0.2 + 230, 0, 255).astype(np.uint8)
    cv2.rectangle(image, (x0, y0), (x1 - 1, y1 - 1), (60, 60, 60), 1)

    inner_x0 = x0 + padding
    inner_y0 = y0 + padding
    inner_x1 = inner_x0 + inset_width
    inner_y1 = inner_y0 + inset_height
    image[inner_y0:inner_y1, inner_x0:inner_x1] = overview_map


def _is_finite_number(value: Optional[float]) -> bool:
    return value is not None and math.isfinite(value) and not math.isnan(value)


def draw_position_marker(
    image: np.ndarray,
    point: tuple[int, int],
    target_edge: int = 34,
    icon_path: str | Path = POSITION_MARKER_ICON_PATH,
) -> None:
    icon = load_position_marker_icon(icon_path)
    if icon is None:
        radius = max(5, int(round(target_edge * 0.22)))
        cv2.circle(image, point, radius, (0, 0, 255), -1, cv2.LINE_AA)
        return

    _overlay_bgra_icon(image, icon, point, target_edge=max(12, int(target_edge)))


def _overlay_bgra_icon(
    image: np.ndarray,
    icon: np.ndarray,
    point: tuple[int, int],
    target_edge: int,
) -> None:
    height, width = icon.shape[:2]
    scale = float(target_edge) / max(height, width, 1)
    if scale != 1.0:
        icon = cv2.resize(
            icon,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
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
