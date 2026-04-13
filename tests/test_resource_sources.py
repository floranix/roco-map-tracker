from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src import resource_sources


def _png_bytes(color_bgr: tuple[int, int, int]) -> bytes:
    image = np.zeros((8, 8, 3), dtype=np.uint8)
    image[:, :] = color_bgr
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("测试图标编码失败")
    return encoded.tobytes()


class ResourceSourcesTestCase(unittest.TestCase):
    def test_resolve_biliwiki_point_projection_matches_current_map_size(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            image = np.full((8704, 12032, 3), 255, dtype=np.uint8)
            map_path = root / "rocom_base_z8.png"
            self.assertTrue(cv2.imwrite(str(map_path), image))

            scale_x, scale_y, offset_x, offset_y = resource_sources._resolve_biliwiki_point_projection(map_path)

        self.assertAlmostEqual(scale_x, 2.0)
        self.assertAlmostEqual(scale_y, 2.0)
        self.assertAlmostEqual(offset_x, 6139.0)
        self.assertAlmostEqual(offset_y, 4335.0)

    def test_build_biliwiki_resource_context_uses_local_icon_cache_dir(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            icon_dir = root / "icon_cache_wiki"
            image = np.full((8704, 12032, 3), 255, dtype=np.uint8)
            map_path = root / "rocom_base_z8.png"
            self.assertTrue(cv2.imwrite(str(map_path), image))

            with patch.object(resource_sources, "WIKI_BASE_MAP_PATH", map_path), patch.object(
                resource_sources, "WIKI_POINTS_PATH", root / "rocom_caiji_points.json"
            ), patch.object(resource_sources, "WIKI_CATEGORIES_PATH", root / "rocom_caiji_categories.json"), patch.object(
                resource_sources, "WIKI_ICON_DIR", icon_dir
            ), patch.object(
                resource_sources,
                "ensure_biliwiki_resource_assets",
                return_value=resource_sources.BiliwikiResourceStatus(),
            ):
                context = resource_sources.build_biliwiki_resource_context()

        self.assertEqual(context.config.poi_icon_dir, str(icon_dir))
        self.assertAlmostEqual(context.config.poi_pixel_scale_x, 2.0)
        self.assertAlmostEqual(context.config.poi_pixel_scale_y, 2.0)
        self.assertAlmostEqual(context.config.poi_pixel_offset_x, 6139.0)
        self.assertAlmostEqual(context.config.poi_pixel_offset_y, 4335.0)

    def test_ensure_biliwiki_resource_assets_caches_collectible_icons(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "rocom_biliwiki"
            base_map_path = data_dir / "rocom_base_z8.png"
            points_path = data_dir / "rocom_caiji_points.json"
            categories_path = data_dir / "rocom_caiji_categories.json"
            icon_dir = data_dir / "icon_cache_wiki"
            icon_bytes = _png_bytes((0, 0, 255))

            def fake_download_file(url: str, path: Path) -> None:
                path.parent.mkdir(parents=True, exist_ok=True)
                if path.suffix == ".json":
                    path.write_text("[]", encoding="utf-8")
                else:
                    path.write_bytes(icon_bytes)

            point_payload = [
                {
                    "id": "ore_1",
                    "markType": 701,
                    "markTypeName": "黑晶琉璃",
                    "point": {"lng": -12, "lat": 34},
                    "iconUrl": "https://example.com/icons/701.png",
                },
                {
                    "id": "plant_1",
                    "markType": 705,
                    "markTypeName": "向阳花",
                    "point": {"lng": 56, "lat": -78},
                    "iconUrl": "https://example.com/icons/705.png",
                },
            ]

            with patch.object(resource_sources, "WIKI_DATA_DIR", data_dir), patch.object(
                resource_sources, "WIKI_BASE_MAP_PATH", base_map_path
            ), patch.object(resource_sources, "WIKI_POINTS_PATH", points_path), patch.object(
                resource_sources, "WIKI_CATEGORIES_PATH", categories_path
            ), patch.object(resource_sources, "WIKI_ICON_DIR", icon_dir), patch.object(
                resource_sources, "_download_file", side_effect=fake_download_file
            ), patch.object(
                resource_sources,
                "_fetch_latest_biliwiki_collectible_points",
                return_value=(point_payload, {701: point_payload[0]["iconUrl"], 705: point_payload[1]["iconUrl"]}),
            ), patch.object(resource_sources, "_download_bytes", return_value=icon_bytes):
                status = resource_sources.ensure_biliwiki_resource_assets(force_refresh_points=True)

            self.assertTrue(status.auto_downloaded_base_map)
            self.assertTrue(status.refreshed_points)
            self.assertTrue((icon_dir / "701.png").exists())
            self.assertTrue((icon_dir / "705.png").exists())

            categories = json.loads(categories_path.read_text(encoding="utf-8"))
            mineral_group = next(group for group in categories if group["title"] == "矿物")
            plant_group = next(group for group in categories if group["title"] == "植物")
            self.assertEqual(mineral_group["categories"][0]["icon"], point_payload[0]["iconUrl"])
            self.assertEqual(plant_group["categories"][0]["icon"], point_payload[1]["iconUrl"])

    def test_existing_points_file_is_not_auto_refreshed_without_force(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "rocom_biliwiki"
            base_map_path = data_dir / "rocom_base_z8.png"
            points_path = data_dir / "rocom_caiji_points.json"
            categories_path = data_dir / "rocom_caiji_categories.json"
            icon_dir = data_dir / "icon_cache_wiki"
            data_dir.mkdir(parents=True, exist_ok=True)
            icon_dir.mkdir(parents=True, exist_ok=True)

            image = np.full((8704, 12032, 3), 255, dtype=np.uint8)
            self.assertTrue(cv2.imwrite(str(base_map_path), image))
            points_path.write_text("[]", encoding="utf-8")

            with patch.object(resource_sources, "WIKI_DATA_DIR", data_dir), patch.object(
                resource_sources, "WIKI_BASE_MAP_PATH", base_map_path
            ), patch.object(resource_sources, "WIKI_POINTS_PATH", points_path), patch.object(
                resource_sources, "WIKI_CATEGORIES_PATH", categories_path
            ), patch.object(resource_sources, "WIKI_ICON_DIR", icon_dir), patch.object(
                resource_sources, "WIKI_METADATA_PATH", data_dir / "metadata.json"
            ), patch.object(
                resource_sources, "_fetch_latest_biliwiki_collectible_points"
            ) as mocked_fetch:
                status = resource_sources.ensure_biliwiki_resource_assets(force_refresh_points=False)

            mocked_fetch.assert_not_called()
            self.assertFalse(status.refreshed_points)


if __name__ == "__main__":
    unittest.main()
