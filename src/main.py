from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import cv2

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.pipeline import LocalizationPipeline
from src.gui import launch_gui
from src.poi_overlay import PoiOverlay, PoiRenderOptions
from src.screen_pick import iterate_screen_region_frames, parse_capture_region
from src.utils import (
    apply_map_metadata_defaults,
    draw_localization,
    ensure_directory,
    guess_poi_categories_path,
    list_image_files,
    load_config,
    load_image,
    result_json,
)


DEFAULT_CONFIG_PATH = "configs/rocom_17173.yaml" if Path("configs/rocom_17173.yaml").exists() else "configs/default.yaml"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="2D 游戏地图定位原型")
    parser.add_argument("--config", default=DEFAULT_CONFIG_PATH, help="YAML 配置文件路径")
    parser.add_argument("--map", dest="map_path", help="覆盖配置中的完整地图路径")
    parser.add_argument("--frame", help="单张局部截图路径")
    parser.add_argument("--frames-dir", help="连续截图文件夹路径")
    parser.add_argument("--video", help="视频文件路径")
    parser.add_argument("--screen-region", help="屏幕采集区域，格式为 x,y,w,h")
    parser.add_argument("--capture-interval-ms", type=int, help="屏幕区域采集间隔（毫秒）")
    parser.add_argument("--output-dir", help="可视化结果输出目录")
    parser.add_argument("--save-visualizations", action="store_true", help="保存可视化结果")
    parser.add_argument("--visualize", action="store_true", help="显示定位窗口")
    parser.add_argument("--gui", action="store_true", help="启动适合 macOS 使用的图形界面")
    parser.add_argument("--poi-data", help="点位数据 JSON 路径")
    parser.add_argument("--poi-categories", help="点位分类 JSON 路径")
    parser.add_argument("--show-poi-overlay", action="store_true", help="在整图上显示筛选后的点位")
    parser.add_argument("--show-poi-labels", action="store_true", help="显示点位名称")
    parser.add_argument("--poi-keyword", help="按标题关键词筛选点位")
    parser.add_argument("--poi-category-ids", help="按分类 ID 过滤，使用逗号分隔")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    if args.gui or not any([args.frame, args.frames_dir, args.video, args.screen_region]):
        launch_gui()
        return 0

    config = load_config(args.config)
    if args.map_path:
        config.map_path = args.map_path
    if args.output_dir:
        config.output_dir = args.output_dir
    if args.save_visualizations:
        config.save_visualizations = True
    if args.poi_data:
        config.poi_data_path = args.poi_data
    if args.poi_categories:
        config.poi_categories_path = args.poi_categories
    if args.show_poi_overlay:
        config.show_poi_overlay = True
    if args.show_poi_labels:
        config.show_poi_labels = True
    if args.poi_keyword is not None:
        config.poi_keyword = args.poi_keyword
    if args.poi_category_ids:
        config.poi_category_ids = parse_category_ids(args.poi_category_ids)
    if args.screen_region:
        config.capture_region = list(parse_capture_region(args.screen_region))
    if args.capture_interval_ms is not None:
        config.capture_interval_ms = max(0, int(args.capture_interval_ms))
    config.poi_categories_path = guess_poi_categories_path(config.poi_data_path, config.poi_categories_path)
    config = apply_map_metadata_defaults(config)

    pipeline = LocalizationPipeline(config)
    output_dir = ensure_directory(config.output_dir) if (config.save_visualizations or args.visualize) else None
    poi_overlay = build_poi_overlay(config)

    for frame_name, frame in iterate_frames(args, config):
        result = pipeline.process_frame(frame)
        print(result_json(frame_name, result))

        if config.save_visualizations or args.visualize:
            map_image = pipeline.map_bundle.color
            extra_lines = None
            if poi_overlay is not None:
                map_image, summary = poi_overlay.render_map(
                    map_image,
                    PoiRenderOptions(
                        enabled=config.show_poi_overlay,
                        selected_category_ids=config.poi_category_ids,
                        keyword=config.poi_keyword,
                        show_labels=config.show_poi_labels,
                        max_points=config.poi_max_draw,
                        label_limit=config.poi_label_limit,
                    ),
                    focus_xy=(result.x, result.y) if not any(map(math.isnan, [result.x, result.y])) else None,
                )
                extra_lines = [summary.text()] if summary is not None else None
            visualization = draw_localization(map_image, frame, result, extra_lines=extra_lines)
            if output_dir is not None:
                output_path = output_dir / f"{frame_name}.png"
                cv2.imwrite(str(output_path), visualization)
            if args.visualize:
                cv2.imshow("定位结果", visualization)
                if cv2.waitKey(1 if (args.video or args.screen_region) else 0) & 0xFF == ord("q"):
                    break

    if args.visualize:
        cv2.destroyAllWindows()
    return 0


def iterate_frames(args, config):
    if args.frame:
        frame_path = Path(args.frame)
        yield frame_path.stem, load_image(frame_path)
        return

    if args.frames_dir:
        for image_path in list_image_files(args.frames_dir):
            yield image_path.stem, load_image(image_path)
        return

    if args.screen_region:
        yield from iterate_screen_region_frames(
            config.capture_region or args.screen_region,
            interval_ms=config.capture_interval_ms,
        )
        return

    capture = cv2.VideoCapture(args.video)
    if not capture.isOpened():
        raise RuntimeError(f"无法打开视频文件：{args.video}")

    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            yield f"frame_{frame_index:06d}", frame
            frame_index += 1
    finally:
        capture.release()


def build_poi_overlay(config):
    if not config.poi_data_path:
        return None
    if config.map_projection == "linear" and len(config.map_bounds) != 4:
        return None

    map_bounds = (
        tuple(float(value) for value in config.map_bounds)
        if len(config.map_bounds) == 4
        else (0.0, 0.0, 1.0, 1.0)
    )
    return PoiOverlay(
        pois_path=config.poi_data_path,
        categories_path=guess_poi_categories_path(config.poi_data_path, config.poi_categories_path) or None,
        map_bounds=map_bounds,
        icon_dir=config.poi_icon_dir or None,
        projection_type=config.map_projection,
        tile_zoom=config.map_tile_zoom,
        tile_x_range=tuple(config.map_tile_x_range) if len(config.map_tile_x_range) == 2 else None,
        tile_y_range=tuple(config.map_tile_y_range) if len(config.map_tile_y_range) == 2 else None,
        tile_size=config.map_tile_size,
        pixel_scale=config.poi_pixel_scale,
        pixel_offset_x=config.poi_pixel_offset_x,
        pixel_offset_y=config.poi_pixel_offset_y,
    )


def parse_category_ids(raw_value: str) -> list[int]:
    values = []
    for item in raw_value.split(","):
        item = item.strip()
        if item:
            values.append(int(item))
    return values


if __name__ == "__main__":
    raise SystemExit(main())
