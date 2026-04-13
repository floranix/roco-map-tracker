from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import ssl
from urllib.request import Request, urlopen

from src.utils import AppConfig, apply_map_metadata_defaults, load_config


RESOURCE_SOURCE_17173 = "17173"
RESOURCE_SOURCE_BILIWIKI = "biliwiki"

RESOURCE_SOURCE_LABELS = {
    RESOURCE_SOURCE_17173: "17173",
    RESOURCE_SOURCE_BILIWIKI: "biliwiki",
}

RESOURCE_SOURCE_DEFAULT = RESOURCE_SOURCE_BILIWIKI

WIKI_DATA_DIR = Path("data/rocom_biliwiki")


@dataclass(frozen=True)
class ResourceSourceContext:
    key: str
    label: str
    config: AppConfig
    source_label: str
    auto_downloaded: bool = False


def resource_source_label(source_key: str) -> str:
    return RESOURCE_SOURCE_LABELS.get(source_key, source_key)


def build_resource_source_context(source_key: str) -> ResourceSourceContext:
    if source_key == RESOURCE_SOURCE_17173:
        config = apply_map_metadata_defaults(load_config("configs/rocom_17173.yaml"))
        return ResourceSourceContext(
            key=RESOURCE_SOURCE_17173,
            label=resource_source_label(RESOURCE_SOURCE_17173),
            config=config,
            source_label="17173",
        )

    if source_key != RESOURCE_SOURCE_BILIWIKI:
        raise ValueError(f"未知资源数据源：{source_key}")

    downloaded = ensure_biliwiki_resource_assets()
    config = AppConfig(
        map_path=str(WIKI_DATA_DIR / "rocom_base_z8.png"),
        display_map_path=str(WIKI_DATA_DIR / "rocom_base_z8.png"),
        map_projection="pixel_space",
        poi_data_path=str(WIKI_DATA_DIR / "rocom_caiji_points.json"),
        poi_categories_path="",
        poi_icon_dir="",
        poi_pixel_scale=1.0,
        poi_pixel_offset_x=3072.0,
        poi_pixel_offset_y=2816.0,
    )
    return ResourceSourceContext(
        key=RESOURCE_SOURCE_BILIWIKI,
        label=resource_source_label(RESOURCE_SOURCE_BILIWIKI),
        config=config,
        source_label="biliwiki",
        auto_downloaded=downloaded,
    )


def ensure_biliwiki_resource_assets() -> bool:
    WIKI_DATA_DIR.mkdir(parents=True, exist_ok=True)
    downloads = {
        WIKI_DATA_DIR / "rocom_base_z8.png": (
            "https://raw.githubusercontent.com/zkjisj/luoke_location/main/out/rocom_base_z8.png"
        ),
        WIKI_DATA_DIR / "rocom_caiji_overlay.png": (
            "https://raw.githubusercontent.com/zkjisj/luoke_location/main/out/rocom_caiji_overlay.png"
        ),
        WIKI_DATA_DIR / "rocom_caiji_points.json": (
            "https://raw.githubusercontent.com/zkjisj/luoke_location/main/out/rocom_caiji_points.json"
        ),
    }

    downloaded = False
    for path, url in downloads.items():
        if path.exists() and path.stat().st_size > 0:
            continue
        _download_file(url, path)
        downloaded = True
    return downloaded


def _download_file(url: str, path: Path) -> None:
    request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    ssl_context = ssl._create_unverified_context()
    try:
        with urlopen(request, timeout=120, context=ssl_context) as response:
            path.write_bytes(response.read())
    except Exception as exc:
        raise RuntimeError(f"下载资源数据失败：{url}") from exc
