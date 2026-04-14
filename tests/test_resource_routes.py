from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.collectible_materials import ACTIVE_COLLECTIBLE_MATERIALS, collectible_ids_for_kind
from src.poi_overlay import PoiCategory, PoiRecord
from src.resource_routes import (
    build_resource_route_plan,
    build_route_cache_signature,
    load_route_plan_cache,
    render_resource_route_plan_viewport,
    route_cache_path,
    save_route_plan_cache,
    summarize_selection_label,
)


class StubOverlay:
    def __init__(self, pois_path: str = "data/test_points.json") -> None:
        self.pois_path = Path(pois_path)

    def project_point(self, longitude: float, latitude: float, image_width: int, image_height: int):
        x = int(round(longitude))
        y = int(round(latitude))
        if x < 0 or y < 0 or x >= image_width or y >= image_height:
            return None
        return x, y


class ResourceRoutesTestCase(unittest.TestCase):
    def test_collectible_material_list_matches_expected_count(self) -> None:
        self.assertEqual(len(ACTIVE_COLLECTIBLE_MATERIALS), 34)
        self.assertEqual(len(collectible_ids_for_kind("mineral")), 4)
        self.assertEqual(len(collectible_ids_for_kind("plant")), 30)

    def test_build_resource_route_plan_filters_by_selected_material_ids(self) -> None:
        overlay = StubOverlay()
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=120, latitude=120),
            PoiRecord(id="ore_b", title="黄石榴石", category_id=702, longitude=165, latitude=120),
            PoiRecord(id="plant_a", title="向阳花", category_id=705, longitude=760, latitude=760),
            PoiRecord(id="plant_b", title="海桑花", category_id=721, longitude=800, latitude=780),
        ]
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿物"),
            702: PoiCategory(id=702, title="黄石榴石", group_id=1, group_title="矿物"),
            705: PoiCategory(id=705, title="向阳花", group_id=2, group_title="植物"),
            721: PoiCategory(id=721, title="海桑花", group_id=2, group_title="植物"),
        }

        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=1024,
            map_height=1024,
            selected_category_ids=[701, 705, 721],
            selection_label="自定义素材",
            start_xy=(100, 100),
            source_label="测试数据",
        )

        self.assertEqual(plan.total_points, 3)
        self.assertEqual(plan.selected_category_ids, [701, 705, 721])
        self.assertEqual(plan.selection_label, "自定义素材")
        self.assertEqual(plan.segments[0].points[0].id, "ore_a")

    def test_route_cache_roundtrip(self) -> None:
        overlay = StubOverlay()
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿物"),
        }
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=120, latitude=120),
            PoiRecord(id="ore_b", title="黑晶琉璃", category_id=701, longitude=180, latitude=120),
        ]
        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=512,
            map_height=512,
            selected_category_ids=[701],
            selection_label="黑晶琉璃",
            source_label="缓存测试",
        )

        signature = build_route_cache_signature(
            selected_category_ids=[701],
            source_path=str(overlay.pois_path),
            source_mtime_ns=123,
            map_path="data/full_map.png",
            map_width=512,
            map_height=512,
            projection_type="pixel_space",
            tile_zoom=0,
            tile_x_range=None,
            tile_y_range=None,
            tile_size=256,
            pixel_scale=1.0,
            pixel_scale_x=0.0,
            pixel_scale_y=0.0,
            pixel_offset_x=0.0,
            pixel_offset_y=0.0,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            cache_path = route_cache_path(temp_dir, signature)
            save_route_plan_cache(cache_path, signature, plan)
            restored = load_route_plan_cache(cache_path, signature)

        self.assertIsNotNone(restored)
        assert restored is not None
        self.assertTrue(restored.cached)
        self.assertEqual(restored.total_points, plan.total_points)
        self.assertEqual(restored.selected_category_ids, [701])

    def test_summarize_selection_label(self) -> None:
        self.assertEqual(summarize_selection_label([701]), "黑晶琉璃")
        self.assertEqual(summarize_selection_label([701, 702, 703]), "黑晶琉璃、黄石榴石、蓝晶碧玺")
        self.assertEqual(summarize_selection_label([701, 702, 703, 704]), "已选 4 种素材")

    def test_render_resource_route_plan_viewport_only_draws_visible_region(self) -> None:
        overlay = StubOverlay()
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿物"),
            705: PoiCategory(id=705, title="向阳花", group_id=2, group_title="植物"),
        }
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=80, latitude=80),
            PoiRecord(id="plant_a", title="向阳花", category_id=705, longitude=120, latitude=120),
        ]
        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=256,
            map_height=256,
            selected_category_ids=[701, 705],
            selection_label="测试素材",
            source_label="测试数据",
        )

        canvas = np.zeros((40, 40, 3), dtype=np.uint8)
        rendered = render_resource_route_plan_viewport(
            canvas,
            plan,
            scale=1.0,
            viewport_origin=(60, 60),
        )
        self.assertGreater(int(rendered.sum()), 0)

    def test_build_resource_route_plan_splits_long_jump_into_multiple_segments(self) -> None:
        overlay = StubOverlay()
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿物"),
        }
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=80, latitude=80),
            PoiRecord(id="ore_b", title="黑晶琉璃", category_id=701, longitude=120, latitude=90),
            PoiRecord(id="ore_c", title="黑晶琉璃", category_id=701, longitude=520, latitude=500),
            PoiRecord(id="ore_d", title="黑晶琉璃", category_id=701, longitude=560, latitude=520),
        ]

        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=1024,
            map_height=1024,
            selected_category_ids=[701],
            selection_label="黑晶琉璃",
            source_label="测试数据",
        )

        self.assertEqual(len(plan.segments), 2)
        self.assertEqual([point.id for point in plan.segments[0].points], ["ore_a", "ore_b"])
        self.assertEqual([point.id for point in plan.segments[1].points], ["ore_c", "ore_d"])

    def test_build_resource_route_plan_keeps_dense_chain_in_single_segment(self) -> None:
        overlay = StubOverlay()
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿物"),
        }
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=80, latitude=80),
            PoiRecord(id="ore_b", title="黑晶琉璃", category_id=701, longitude=140, latitude=85),
            PoiRecord(id="ore_c", title="黑晶琉璃", category_id=701, longitude=205, latitude=88),
            PoiRecord(id="ore_d", title="黑晶琉璃", category_id=701, longitude=270, latitude=92),
        ]

        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=1024,
            map_height=1024,
            selected_category_ids=[701],
            selection_label="黑晶琉璃",
            source_label="测试数据",
        )

        self.assertEqual(len(plan.segments), 1)
        self.assertEqual(len(plan.segments[0].points), 4)


if __name__ == "__main__":
    unittest.main()
