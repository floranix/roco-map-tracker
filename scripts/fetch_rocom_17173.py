from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import math
import re
import ssl
from pathlib import Path
from urllib.request import Request, urlopen

import cv2
import numpy as np


SOURCE_PAGE = "https://map.17173.com/rocom/maps/shijie"
DEFAULT_OUTPUT_DIR = Path("data/rocom_17173")
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/146.0.0.0 Safari/537.36"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="拉取 17173 洛克王国世界互动地图数据")
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="输出目录，默认写入 data/rocom_17173",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    html = fetch_text(SOURCE_PAGE)
    write_text(output_dir / "page.html", html)

    bootstrap_url = build_bootstrap_url()
    bootstrap_js = fetch_text(bootstrap_url)
    write_text(output_dir / "bootstrap.js", bootstrap_js)

    bundle_url = require_match(
        r"https://ue\.17173cdn\.com/a/terra/web/assets/index-[A-Za-z0-9_-]+\.js",
        bootstrap_js,
        "前端 bundle URL",
    )
    bundle_js = fetch_text(bundle_url)
    write_text(output_dir / "bundle.js", bundle_js)

    map_id = int(require_match(r"1731003:\{shijie:(\d+)\}", bundle_js, "map_id"))
    version = require_match(
        rf"{map_id}:\{{id:{map_id},.*?version:\"([^\"]+)\"",
        bundle_js,
        "地图版本号",
    )
    min_zoom = int(
        require_match(
            rf"{map_id}:\{{id:{map_id},.*?minZoom:(\d+)",
            bundle_js,
            "最小缩放",
        )
    )
    max_zoom = int(
        require_match(
            rf"{map_id}:\{{id:{map_id},.*?maxZoom:(\d+)",
            bundle_js,
            "最大缩放",
        )
    )
    initial_zoom = int(
        require_match(
            rf"{map_id}:\{{id:{map_id},.*?initialZoom:(\d+)",
            bundle_js,
            "默认缩放",
        )
    )
    map_maxzoom = int(
        require_match(
            rf"{map_id}:\{{id:{map_id},.*?mapMaxzoom:(\d+)",
            bundle_js,
            "瓦片最大缩放",
        )
    )
    bounds_text = require_match(
        rf"{map_id}:\{{id:{map_id},.*?bounds:\[([^\]]+)\]",
        bundle_js,
        "地图边界",
    )
    bounds = [float(item.strip()) for item in bounds_text.split(",")]
    cover_url = require_match(
        r"https://ue\.17173cdn\.com/a/terra/web/assets/rocom-shijie\.(?:jpg|png|webp)",
        bundle_js,
        "封面图 URL",
    )
    tile_template = (
        f"https://ue.17173cdn.com/a/terra/tiles/rocom/{map_id}_{version}/{{z}}/{{y}}_{{x}}.png?v1"
    )

    cover_bytes = fetch_binary(cover_url)
    cover_path = output_dir / Path(cover_url).name
    cover_path.write_bytes(cover_bytes)

    tile_grid = build_tile_grid(bounds=bounds, zoom=map_maxzoom)
    tile_dir = output_dir / f"tiles_z{map_maxzoom}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    download_tiles(
        tile_template=tile_template,
        zoom=map_maxzoom,
        x_min=tile_grid["x_min"],
        x_max=tile_grid["x_max"],
        y_min=tile_grid["y_min"],
        y_max=tile_grid["y_max"],
        output_dir=tile_dir,
    )
    stitched_map_path = output_dir / f"{cover_path.stem}-z{map_maxzoom}.png"
    stitch_tiles(
        tile_dir=tile_dir,
        x_min=tile_grid["x_min"],
        x_max=tile_grid["x_max"],
        y_min=tile_grid["y_min"],
        y_max=tile_grid["y_max"],
        output_path=stitched_map_path,
        tile_size=tile_grid["tile_size"],
    )

    pois_url = f"https://terra-api.17173.com/app/location/list?mapIds={map_id}"
    regions_url = f"https://terra-api.17173.com/app/region/list?mapId={map_id}"

    pois_payload = fetch_json(pois_url)
    regions_payload = fetch_json(regions_url)
    save_json(output_dir / "pois.json", pois_payload)
    save_json(output_dir / "regions.json", regions_payload)
    categories_payload = extract_rocom_categories(bundle_js, map_id=map_id, game_id=1731003)
    save_json(output_dir / "categories.json", categories_payload)

    metadata = {
        "source_page": SOURCE_PAGE,
        "bootstrap_url": bootstrap_url,
        "bundle_url": bundle_url,
        "game_id": 1731003,
        "game_key": "rocom",
        "game_title": "洛克王国世界",
        "map_name": "shijie",
        "map_id": map_id,
        "version": version,
        "min_zoom": min_zoom,
        "max_zoom": max_zoom,
        "initial_zoom": initial_zoom,
        "map_maxzoom": map_maxzoom,
        "bounds": bounds,
        "cover_url": cover_url,
        "cover_file": str(cover_path.name),
        "stitched_map_file": str(stitched_map_path.name),
        "stitched_map_projection": "web_mercator_tiles",
        "tile_size": tile_grid["tile_size"],
        "max_zoom_tile_grid": tile_grid,
        "pois_api": pois_url,
        "regions_api": regions_url,
        "tile_url_template": tile_template,
        "pois_count": count_items(pois_payload),
        "regions_count": count_items(regions_payload),
        "category_group_count": len(categories_payload),
        "category_count": sum(len(group["categories"]) for group in categories_payload),
        "notes": [
            "pois.json 是互动地图点位原始数据，可直接用于原型期的坐标验证。",
            "tile_url_template 来自前端 bundle 里的地图初始化逻辑。",
            "categories.json 提供点位分类与图标信息，可用于按需筛选显示。",
            "当前脚本只验证并保存封面图、点位和区域数据，不主动穷举下载底图瓦片。",
        ],
    }
    save_json(output_dir / "metadata.json", metadata)

    print(f"已写入: {output_dir}")
    print(f"map_id={map_id}, version={version}")
    print(f"pois={metadata['pois_count']}, regions={metadata['regions_count']}")
    print(f"stitched_map={stitched_map_path.name} ({tile_grid['pixel_width']}x{tile_grid['pixel_height']})")
    print(f"tile_url_template={tile_template}")
    return 0


def build_tile_grid(bounds: list[float], zoom: int, tile_size: int = 256) -> dict[str, int]:
    min_lon, min_lat, max_lon, max_lat = bounds
    x_min = clamp_tile_index(math.floor(lon_to_tile_x(min_lon, zoom)), zoom)
    x_max = clamp_tile_index(math.ceil(lon_to_tile_x(max_lon, zoom)) - 1, zoom)
    y_min = clamp_tile_index(math.floor(lat_to_tile_y(max_lat, zoom)), zoom)
    y_max = clamp_tile_index(math.ceil(lat_to_tile_y(min_lat, zoom)) - 1, zoom)

    return {
        "zoom": zoom,
        "tile_size": tile_size,
        "x_min": x_min,
        "x_max": x_max,
        "y_min": y_min,
        "y_max": y_max,
        "tile_columns": x_max - x_min + 1,
        "tile_rows": y_max - y_min + 1,
        "pixel_width": (x_max - x_min + 1) * tile_size,
        "pixel_height": (y_max - y_min + 1) * tile_size,
    }


def lon_to_tile_x(longitude: float, zoom: int) -> float:
    return (longitude + 180.0) / 360.0 * (2**zoom)


def lat_to_tile_y(latitude: float, zoom: int) -> float:
    latitude = max(min(latitude, 85.05112878), -85.05112878)
    latitude_rad = math.radians(latitude)
    return (1.0 - math.asinh(math.tan(latitude_rad)) / math.pi) / 2.0 * (2**zoom)


def clamp_tile_index(index: int, zoom: int) -> int:
    max_index = 2**zoom - 1
    return max(0, min(max_index, index))


def download_tiles(
    tile_template: str,
    zoom: int,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    output_dir: Path,
) -> None:
    coordinates = [(x, y) for y in range(y_min, y_max + 1) for x in range(x_min, x_max + 1)]
    errors = []

    def fetch_one(coord: tuple[int, int]) -> tuple[int, int]:
        x, y = coord
        tile_path = output_dir / f"{y}_{x}.png"
        if tile_path.exists():
            return coord

        tile_url = tile_template.format(z=zoom, y=y, x=x)
        tile_bytes = fetch_binary(tile_url)
        tile_path.write_bytes(tile_bytes)
        return coord

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_one, coord): coord for coord in coordinates}
        for future in as_completed(futures):
            coord = futures[future]
            try:
                future.result()
            except Exception as exc:  # pragma: no cover - depends on remote host
                errors.append((coord, exc))

    if errors:
        sample = ", ".join(f"{x}/{y}: {exc}" for (x, y), exc in errors[:5])
        raise RuntimeError(f"下载瓦片失败，共 {len(errors)} 个。样例：{sample}")


def stitch_tiles(
    tile_dir: Path,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    output_path: Path,
    tile_size: int,
) -> None:
    width = (x_max - x_min + 1) * tile_size
    height = (y_max - y_min + 1) * tile_size
    canvas = np.zeros((height, width, 3), dtype=np.uint8)

    for y in range(y_min, y_max + 1):
        for x in range(x_min, x_max + 1):
            tile_path = tile_dir / f"{y}_{x}.png"
            tile = cv2.imread(str(tile_path), cv2.IMREAD_UNCHANGED)
            if tile is None:
                raise RuntimeError(f"读取瓦片失败：{tile_path}")

            if tile.ndim == 2:
                tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
            elif tile.shape[2] == 4:
                alpha = tile[:, :, 3:4].astype(np.float32) / 255.0
                rgb = tile[:, :, :3].astype(np.float32)
                tile = np.uint8(rgb * alpha + 255.0 * (1.0 - alpha))

            offset_x = (x - x_min) * tile_size
            offset_y = (y - y_min) * tile_size
            canvas[offset_y : offset_y + tile_size, offset_x : offset_x + tile_size] = tile[:, :, :3]

    if not cv2.imwrite(str(output_path), canvas):
        raise RuntimeError(f"写入拼接地图失败：{output_path}")


def extract_rocom_categories(bundle_js: str, map_id: int, game_id: int) -> list[dict]:
    anchor = f"{map_id}:["
    anchor_index = bundle_js.find(anchor)
    if anchor_index < 0:
        raise RuntimeError("未找到 rocom 分类段")

    section = extract_balanced_block(bundle_js, anchor_index + len(f"{map_id}:"), "[", "]")
    group_pattern = re.compile(
        rf"\{{game_id:{game_id},title:\"([^\"]+)\",id:(\d+),categories:\[(.*?)\]\}}",
        re.DOTALL,
    )
    category_pattern = re.compile(
        r"\{title:\"([^\"]+)\",group_id:(\d+),id:(\d+),icon:\"([^\"]+)\"\}"
    )

    groups = []
    for group_match in group_pattern.finditer(section):
        group_title, group_id, categories_blob = group_match.groups()
        categories = []
        for category_match in category_pattern.finditer(categories_blob):
            title, category_group_id, category_id, icon = category_match.groups()
            categories.append(
                {
                    "id": int(category_id),
                    "title": title,
                    "group_id": int(category_group_id),
                    "icon": icon,
                }
            )
        groups.append(
            {
                "id": int(group_id),
                "title": group_title,
                "categories": categories,
            }
        )

    if not groups:
        raise RuntimeError("未能解析出 rocom 点位分类")
    return groups


def extract_balanced_block(text: str, start_index: int, open_char: str, close_char: str) -> str:
    depth = 0
    in_string = False
    escaped = False
    buffer = []

    for index in range(start_index, len(text)):
        char = text[index]
        buffer.append(char)

        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
            continue

        if char == open_char:
            depth += 1
        elif char == close_char:
            depth -= 1
            if depth == 0:
                return "".join(buffer)

    raise RuntimeError("未能提取平衡块")


def build_bootstrap_url() -> str:
    return "https://ue.17173cdn.com/a/terra/web/bootstrap.js"


def fetch_text(url: str) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": SOURCE_PAGE})
    with urlopen(request, context=ssl._create_unverified_context(), timeout=30) as response:
        return response.read().decode("utf-8", errors="ignore")


def fetch_binary(url: str) -> bytes:
    request = Request(url, headers={"User-Agent": USER_AGENT, "Referer": SOURCE_PAGE})
    with urlopen(request, context=ssl._create_unverified_context(), timeout=30) as response:
        return response.read()


def fetch_json(url: str):
    text = fetch_text(url)
    return json.loads(text)


def count_items(payload) -> int:
    if isinstance(payload, list):
        return len(payload)
    if isinstance(payload, dict) and isinstance(payload.get("data"), list):
        return len(payload["data"])
    return 0


def require_match(pattern: str, text: str, label: str) -> str:
    match = re.search(pattern, text, re.DOTALL)
    if not match:
        raise RuntimeError(f"未找到 {label}")
    return match.group(1) if match.groups() else match.group(0)


def write_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8")


def save_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
