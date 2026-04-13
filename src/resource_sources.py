from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
from pathlib import Path
import ssl
import time
from urllib.parse import urlparse
from urllib.request import Request, urlopen

import cv2

from src.collectible_materials import (
    ACTIVE_COLLECTIBLE_IDS,
    ACTIVE_COLLECTIBLE_MATERIALS,
    COLLECTIBLE_KIND_MINERAL,
    COLLECTIBLE_KIND_PLANT,
)
from src.utils import AppConfig


RESOURCE_SOURCE_BILIWIKI = "biliwiki"
WIKI_DATA_DIR = Path("data/rocom_biliwiki")
WIKI_BASE_MAP_PATH = WIKI_DATA_DIR / "rocom_base_z8.png"
WIKI_POINTS_PATH = WIKI_DATA_DIR / "rocom_caiji_points.json"
WIKI_CATEGORIES_PATH = WIKI_DATA_DIR / "rocom_caiji_categories.json"
WIKI_ICON_DIR = WIKI_DATA_DIR / "icon_cache_wiki"
WIKI_METADATA_PATH = WIKI_DATA_DIR / "metadata.json"

WIKI_MAP_PAGE_URL = "https://wiki.biligame.com/rocom/%E5%A4%A7%E5%9C%B0%E5%9B%BE"
WIKI_POINT_PAGE_URL = "https://wiki.biligame.com/rocom/Data:Mapnew/point.json"
BASE_MAP_FALLBACK_URL = "https://raw.githubusercontent.com/zkjisj/luoke_location/main/out/rocom_base_z8.png"
POINTS_FALLBACK_URL = "https://raw.githubusercontent.com/zkjisj/luoke_location/main/out/rocom_caiji_points.json"
BILIWIKI_REFERENCE_MAP_WIDTH = 12032.0
BILIWIKI_REFERENCE_MAP_HEIGHT = 8704.0
BILIWIKI_REFERENCE_POINT_SCALE_X = 2.0
BILIWIKI_REFERENCE_POINT_SCALE_Y = 2.0
BILIWIKI_REFERENCE_POINT_OFFSET_X = 6139.0
BILIWIKI_REFERENCE_POINT_OFFSET_Y = 4335.0


@dataclass(frozen=True)
class BiliwikiResourceStatus:
    auto_downloaded_base_map: bool = False
    refreshed_points: bool = False


@dataclass(frozen=True)
class ResourceSourceContext:
    key: str
    label: str
    config: AppConfig
    source_label: str
    auto_downloaded: bool = False
    points_refreshed: bool = False


def build_biliwiki_resource_context(force_refresh_points: bool = False) -> ResourceSourceContext:
    status = ensure_biliwiki_resource_assets(force_refresh_points=force_refresh_points)
    scale_x, scale_y, offset_x, offset_y = _resolve_biliwiki_point_projection(WIKI_BASE_MAP_PATH)
    config = AppConfig(
        map_path=str(WIKI_BASE_MAP_PATH),
        display_map_path=str(WIKI_BASE_MAP_PATH),
        map_projection="pixel_space",
        poi_data_path=str(WIKI_POINTS_PATH),
        poi_categories_path=str(WIKI_CATEGORIES_PATH),
        poi_icon_dir=str(WIKI_ICON_DIR),
        poi_pixel_scale=1.0,
        poi_pixel_scale_x=scale_x,
        poi_pixel_scale_y=scale_y,
        poi_pixel_offset_x=offset_x,
        poi_pixel_offset_y=offset_y,
    )
    return ResourceSourceContext(
        key=RESOURCE_SOURCE_BILIWIKI,
        label=RESOURCE_SOURCE_BILIWIKI,
        config=config,
        source_label="biliwiki 采集",
        auto_downloaded=status.auto_downloaded_base_map,
        points_refreshed=status.refreshed_points,
    )


def ensure_biliwiki_resource_assets(force_refresh_points: bool = False) -> BiliwikiResourceStatus:
    WIKI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    WIKI_ICON_DIR.mkdir(parents=True, exist_ok=True)

    auto_downloaded_base_map = False
    if not WIKI_BASE_MAP_PATH.exists() or WIKI_BASE_MAP_PATH.stat().st_size <= 0:
        _download_file(BASE_MAP_FALLBACK_URL, WIKI_BASE_MAP_PATH)
        auto_downloaded_base_map = True

    refreshed_points = False
    icon_urls: dict[int, str] = {}
    try:
        if force_refresh_points or not WIKI_POINTS_PATH.exists() or WIKI_POINTS_PATH.stat().st_size <= 0:
            points_payload, icon_urls = _fetch_latest_biliwiki_collectible_points()
            WIKI_POINTS_PATH.write_text(
                json.dumps(points_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            refreshed_points = True
    except Exception:
        if not WIKI_POINTS_PATH.exists() or WIKI_POINTS_PATH.stat().st_size <= 0:
            _download_file(POINTS_FALLBACK_URL, WIKI_POINTS_PATH)
            refreshed_points = True

    if not icon_urls:
        icon_urls = _collect_icon_urls_from_points_file(WIKI_POINTS_PATH)
    _write_collectible_categories(icon_urls)
    _sync_collectible_icon_cache(icon_urls, force_refresh=force_refresh_points or refreshed_points)
    scale_x, scale_y, offset_x, offset_y = _resolve_biliwiki_point_projection(WIKI_BASE_MAP_PATH)
    _write_biliwiki_metadata(scale_x, scale_y, offset_x, offset_y)

    return BiliwikiResourceStatus(
        auto_downloaded_base_map=auto_downloaded_base_map,
        refreshed_points=refreshed_points,
    )


def apply_biliwiki_resource_defaults(
    config: AppConfig,
    force_refresh_points: bool = False,
) -> AppConfig:
    context = build_biliwiki_resource_context(force_refresh_points=force_refresh_points)
    builtin = context.config

    map_path = Path(config.map_path).expanduser() if str(config.map_path).strip() else None
    if map_path is None or not map_path.exists() or map_path == Path("data/full_map.png"):
        config.map_path = builtin.map_path

    if not config.display_map_path:
        config.display_map_path = builtin.display_map_path or builtin.map_path

    if not config.poi_data_path:
        config.poi_data_path = builtin.poi_data_path
    if not config.poi_categories_path:
        config.poi_categories_path = builtin.poi_categories_path
    if not config.poi_icon_dir:
        config.poi_icon_dir = builtin.poi_icon_dir

    if config.map_projection in {"", "linear"}:
        config.map_projection = builtin.map_projection
    if not config.map_tile_zoom:
        config.map_tile_zoom = builtin.map_tile_zoom
    if not config.map_tile_x_range:
        config.map_tile_x_range = list(builtin.map_tile_x_range)
    if not config.map_tile_y_range:
        config.map_tile_y_range = list(builtin.map_tile_y_range)
    if not config.map_tile_size:
        config.map_tile_size = builtin.map_tile_size

    if config.poi_pixel_scale == 1.0:
        config.poi_pixel_scale = builtin.poi_pixel_scale
    if config.poi_pixel_scale_x == 0.0:
        config.poi_pixel_scale_x = builtin.poi_pixel_scale_x
    if config.poi_pixel_scale_y == 0.0:
        config.poi_pixel_scale_y = builtin.poi_pixel_scale_y
    if config.poi_pixel_offset_x == 0.0:
        config.poi_pixel_offset_x = builtin.poi_pixel_offset_x
    if config.poi_pixel_offset_y == 0.0:
        config.poi_pixel_offset_y = builtin.poi_pixel_offset_y
    return config


def _fetch_latest_biliwiki_collectible_points() -> tuple[list[dict], dict[int, str]]:
    category_html = _download_text(WIKI_MAP_PAGE_URL)
    point_html = _download_text(WIKI_POINT_PAGE_URL)

    category_metadata = _parse_collectible_category_metadata(category_html)
    point_payload = _parse_collectible_point_payload(point_html)

    points: list[dict] = []
    icon_urls: dict[int, str] = {}
    for material in ACTIVE_COLLECTIBLE_MATERIALS:
        metadata = category_metadata.get(material.category_id, {})
        icon_url = str(metadata.get("icon_url") or "")
        icon_urls[material.category_id] = icon_url

        for index, raw_point in enumerate(point_payload.get(material.category_id, []), start=1):
            point = dict(raw_point)
            point["markType"] = material.category_id
            point["markTypeName"] = material.name
            point["title"] = str(point.get("title") or "")
            if icon_url and not point.get("iconUrl"):
                point["iconUrl"] = icon_url
            if not point.get("id"):
                point["id"] = f"{material.category_id}_{index}"
            points.append(point)

    points.sort(
        key=lambda item: (
            int(item.get("markType") or 0),
            float((item.get("point") or {}).get("lat") or 0.0),
            float((item.get("point") or {}).get("lng") or 0.0),
            str(item.get("id") or ""),
        )
    )
    return points, icon_urls


def _parse_collectible_category_metadata(page_html: str) -> dict[int, dict[str, str]]:
    match = re.search(r'id="categoryData"[^>]*>(.*?)</div>', page_html, re.S)
    if match is None:
        return {}

    chunk = match.group(1).replace("<p>", "").replace("</p>", "")
    chunk = re.sub(r'<a [^>]*href="([^"]+)"[^>]*>.*?</a>', r"\1", chunk, flags=re.S)
    payload = json.loads(html.unescape(chunk))

    metadata: dict[int, dict[str, str]] = {}
    for item in payload.get("data", []):
        if str(item.get("type") or "") != "采集":
            continue
        category_id = int(item.get("markType") or 0)
        if category_id not in ACTIVE_COLLECTIBLE_IDS:
            continue
        metadata[category_id] = {
            "name": str(item.get("markTypeName") or "").strip(),
            "icon_url": str(item.get("icon") or "").strip(),
        }
    return metadata


def _parse_collectible_point_payload(page_html: str) -> dict[int, list[dict]]:
    match = re.search(r'id="mapPointData"[^>]*>(.*?)</div>', page_html, re.S)
    if match is None:
        return {}

    chunk = html.unescape(match.group(1).replace("<p>", "").replace("</p>", ""))
    payload: dict[int, list[dict]] = {}
    for category_id in ACTIVE_COLLECTIBLE_IDS:
        pattern = rf"\b{category_id}\s*:\s*(\[[\s\S]*?\])(?=\s*,\s*\d+\s*:|\s*\}})"
        category_match = re.search(pattern, chunk)
        if category_match is None:
            payload[category_id] = []
            continue
        payload[category_id] = json.loads(category_match.group(1))
    return payload


def _collect_icon_urls_from_points_file(path: Path) -> dict[int, str]:
    if not path.exists():
        return {}

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    raw_records = payload.get("data") if isinstance(payload, dict) else payload
    if not isinstance(raw_records, list):
        return {}

    icon_urls: dict[int, str] = {}
    for item in raw_records:
        try:
            category_id = int(item.get("markType") or 0)
        except (AttributeError, TypeError, ValueError):
            continue
        if category_id not in ACTIVE_COLLECTIBLE_IDS:
            continue
        icon_url = str(item.get("iconUrl") or "").strip()
        if icon_url and category_id not in icon_urls:
            icon_urls[category_id] = icon_url
    return icon_urls


def _write_collectible_categories(icon_urls: dict[int, str]) -> None:
    categories = []
    for group_id, (group_title, kind) in enumerate(
        (("矿物", COLLECTIBLE_KIND_MINERAL), ("植物", COLLECTIBLE_KIND_PLANT)),
        start=1,
    ):
        group_categories = []
        for material in ACTIVE_COLLECTIBLE_MATERIALS:
            if material.kind != kind:
                continue
            group_categories.append(
                {
                    "id": material.category_id,
                    "title": material.name,
                    "group_id": group_id,
                    "icon": icon_urls.get(material.category_id, ""),
                }
            )
        categories.append(
            {
                "id": group_id,
                "title": group_title,
                "categories": group_categories,
            }
        )

    WIKI_CATEGORIES_PATH.write_text(
        json.dumps(categories, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _sync_collectible_icon_cache(icon_urls: dict[int, str], force_refresh: bool = False) -> None:
    WIKI_ICON_DIR.mkdir(parents=True, exist_ok=True)

    for category_id, icon_url in sorted(icon_urls.items()):
        if category_id not in ACTIVE_COLLECTIBLE_IDS:
            continue

        icon_url = str(icon_url or "").strip()
        if not icon_url:
            continue

        parsed = urlparse(icon_url)
        suffix = Path(parsed.path).suffix.lower() or ".png"
        target_path = WIKI_ICON_DIR / f"{category_id}{suffix}"

        if not force_refresh and target_path.exists() and target_path.stat().st_size > 0:
            continue

        for existing_path in WIKI_ICON_DIR.glob(f"{category_id}.*"):
            if existing_path != target_path:
                existing_path.unlink(missing_ok=True)

        if parsed.scheme in {"http", "https"}:
            target_path.write_bytes(_download_bytes(icon_url))
            continue

        local_path = Path(icon_url).expanduser()
        if local_path.exists() and local_path.is_file():
            target_path.write_bytes(local_path.read_bytes())


def _resolve_biliwiki_point_projection(map_path: Path) -> tuple[float, float, float, float]:
    map_image = cv2.imread(str(map_path), cv2.IMREAD_COLOR)
    if map_image is None:
        raise RuntimeError(f"读取 biliwiki 底图失败：{map_path}")

    image_height, image_width = map_image.shape[:2]
    width_ratio = image_width / BILIWIKI_REFERENCE_MAP_WIDTH
    height_ratio = image_height / BILIWIKI_REFERENCE_MAP_HEIGHT
    scale_x = BILIWIKI_REFERENCE_POINT_SCALE_X * width_ratio
    scale_y = BILIWIKI_REFERENCE_POINT_SCALE_Y * height_ratio
    offset_x = BILIWIKI_REFERENCE_POINT_OFFSET_X * width_ratio
    offset_y = BILIWIKI_REFERENCE_POINT_OFFSET_Y * height_ratio
    return scale_x, scale_y, offset_x, offset_y


def _write_biliwiki_metadata(
    scale_x: float,
    scale_y: float,
    offset_x: float,
    offset_y: float,
) -> None:
    payload = {
        "source_page": WIKI_MAP_PAGE_URL,
        "points_source": WIKI_POINT_PAGE_URL,
        "stitched_map_file": WIKI_BASE_MAP_PATH.name,
        "stitched_map_projection": "pixel_space",
        "poi_icon_dir": WIKI_ICON_DIR.name,
        "poi_pixel_scale": 1.0,
        "poi_pixel_scale_x": float(scale_x),
        "poi_pixel_scale_y": float(scale_y),
        "poi_pixel_offset_x": float(offset_x),
        "poi_pixel_offset_y": float(offset_y),
        "updated_at": int(time.time()),
    }
    WIKI_METADATA_PATH.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _download_file(url: str, path: Path) -> None:
    path.write_text(_download_text(url), encoding="utf-8") if path.suffix == ".json" else path.write_bytes(
        _download_bytes(url)
    )


def _download_text(url: str) -> str:
    return _download_bytes(url).decode("utf-8", errors="ignore")


def _download_bytes(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ssl_context = ssl._create_unverified_context()
    with urlopen(request, timeout=120, context=ssl_context) as response:
        return response.read()
