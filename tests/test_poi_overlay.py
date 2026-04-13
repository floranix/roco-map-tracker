from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.poi_overlay import PoiOverlay, PoiRenderOptions


class PoiOverlayTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        (self.root / "pois.json").write_text(
            json.dumps(
                {
                    "data": [
                        {"id": 1, "title": "宝箱 A", "category_id": 1001, "longitude": -1.4, "latitude": 1.4},
                        {"id": 2, "title": "宝箱 B", "category_id": 1001, "longitude": -0.7, "latitude": 0.7},
                        {"id": 3, "title": "矿石 C", "category_id": 1002, "longitude": 0.0, "latitude": 0.0},
                    ]
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        (self.root / "categories.json").write_text(
            json.dumps(
                [
                    {
                        "id": 10,
                        "title": "收集",
                        "categories": [
                            {"id": 1001, "title": "宝箱", "group_id": 10, "icon": ""},
                            {"id": 1002, "title": "矿石", "group_id": 10, "icon": ""},
                        ],
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_project_point_uses_linear_bounds_mapping(self) -> None:
        overlay = PoiOverlay(
            pois_path=self.root / "pois.json",
            categories_path=self.root / "categories.json",
            map_bounds=(-1.4, 0.0, 0.0, 1.4),
        )

        self.assertEqual(overlay.project_point(-1.4, 1.4, 140, 140), (0, 0))
        self.assertEqual(overlay.project_point(-0.7, 0.7, 140, 140), (70, 70))
        self.assertEqual(overlay.project_point(0.0, 0.0, 140, 140), (139, 139))

    def test_render_map_requires_selected_categories(self) -> None:
        overlay = PoiOverlay(
            pois_path=self.root / "pois.json",
            categories_path=self.root / "categories.json",
            map_bounds=(-1.4, 0.0, 0.0, 1.4),
        )
        image = np.zeros((140, 140, 3), dtype=np.uint8)

        rendered, summary = overlay.render_map(
            image,
            PoiRenderOptions(
                enabled=True,
                selected_category_ids=[],
                keyword="",
                show_labels=False,
            ),
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.matched, 0)
        self.assertEqual(summary.rendered, 0)
        self.assertEqual(summary.note, "未选分类")
        self.assertEqual(int(rendered.sum()), 0)

    def test_render_map_filters_by_category_and_keyword(self) -> None:
        overlay = PoiOverlay(
            pois_path=self.root / "pois.json",
            categories_path=self.root / "categories.json",
            map_bounds=(-1.4, 0.0, 0.0, 1.4),
        )
        image = np.zeros((140, 140, 3), dtype=np.uint8)

        rendered, summary = overlay.render_map(
            image,
            PoiRenderOptions(
                enabled=True,
                selected_category_ids=[1001],
                keyword="B",
                show_labels=False,
            ),
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.total, 3)
        self.assertEqual(summary.matched, 1)
        self.assertEqual(summary.rendered, 1)
        self.assertGreater(int(rendered.sum()), 0)

    def test_project_point_supports_web_mercator_tiles(self) -> None:
        overlay = PoiOverlay(
            pois_path=self.root / "pois.json",
            categories_path=self.root / "categories.json",
            map_bounds=(-180.0, -85.0, 180.0, 85.0),
            projection_type="web_mercator_tiles",
            tile_zoom=1,
            tile_x_range=(0, 1),
            tile_y_range=(0, 1),
            tile_size=256,
        )

        self.assertEqual(overlay.project_point(-180.0, 85.05112878, 512, 512), (0, 0))
        self.assertEqual(overlay.project_point(0.0, 0.0, 512, 512), (256, 256))
        self.assertEqual(overlay.project_point(180.0, -85.05112878, 512, 512), (511, 511))

    def test_project_point_supports_pixel_space_projection(self) -> None:
        overlay = PoiOverlay(
            pois_path=self.root / "pois.json",
            map_bounds=(0.0, 0.0, 1.0, 1.0),
            projection_type="pixel_space",
            pixel_scale=2.0,
            pixel_offset_x=5.0,
            pixel_offset_y=7.0,
        )

        self.assertEqual(overlay.project_point(10.0, 20.0, 80, 80), (25, 47))
        self.assertIsNone(overlay.project_point(40.0, 40.0, 80, 80))

    def test_biligame_style_records_use_icon_cache_and_mark_type_name(self) -> None:
        (self.root / "biligame_points.json").write_text(
            json.dumps(
                [
                    {
                        "id": "abc",
                        "markType": 201,
                        "markTypeName": "眠枭庇护所",
                        "point": {"lng": 12, "lat": 18},
                        "iconUrl": "https://example.com/icon.png",
                    }
                ],
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )
        icon_dir = self.root / "icon_cache_wiki"
        icon_dir.mkdir()
        icon = np.zeros((12, 12, 4), dtype=np.uint8)
        icon[:, :, 2] = 255
        icon[:, :, 3] = 255
        self.assertTrue(cv2.imwrite(str(icon_dir / "201_test.png"), icon))

        overlay = PoiOverlay(
            pois_path=self.root / "biligame_points.json",
            map_bounds=(0.0, 0.0, 1.0, 1.0),
            icon_dir=icon_dir,
            projection_type="pixel_space",
        )
        categories = overlay.available_categories()
        self.assertEqual(categories[0][0].title, "眠枭庇护所")

        rendered, summary = overlay.render_map(
            np.zeros((40, 40, 3), dtype=np.uint8),
            PoiRenderOptions(
                enabled=True,
                selected_category_ids=[201],
            ),
        )

        self.assertIsNotNone(summary)
        self.assertEqual(summary.rendered, 1)
        self.assertGreater(int(rendered[12:25, 6:19].sum()), 0)


if __name__ == "__main__":
    unittest.main()
