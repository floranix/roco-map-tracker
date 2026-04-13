from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.poi_overlay import PoiCategory, PoiRecord
from src.resource_routes import (
    RESOURCE_ROUTE_MODE_ORE,
    RESOURCE_ROUTE_MODE_ORE_AND_PLANT,
    RESOURCE_ROUTE_MODE_PLANT,
    build_resource_route_plan,
    build_route_cache_signature,
    infer_resource_kind,
    infer_resource_kind_from_texts,
    load_route_plan_cache,
    route_cache_path,
    save_route_plan_cache,
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
    def test_infer_resource_kind_recognizes_ore_and_plant_categories(self) -> None:
        ore_category = PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿石")
        plant_category = PoiCategory(id=705, title="向阳花", group_id=2, group_title="花草")
        categories = {
            ore_category.id: ore_category,
            plant_category.id: plant_category,
        }

        ore_record = PoiRecord(
            id="ore",
            title="黑晶琉璃",
            category_id=701,
            longitude=100,
            latitude=100,
        )
        plant_record = PoiRecord(
            id="plant",
            title="向阳花",
            category_id=705,
            longitude=120,
            latitude=100,
        )

        self.assertEqual(infer_resource_kind(ore_record, categories), RESOURCE_ROUTE_MODE_ORE)
        self.assertEqual(infer_resource_kind(plant_record, categories), RESOURCE_ROUTE_MODE_PLANT)
        self.assertEqual(
            infer_resource_kind_from_texts(
                title="无花果树",
                category_title="",
                resolved_category_title="",
                group_title="",
            ),
            RESOURCE_ROUTE_MODE_PLANT,
        )

    def test_build_resource_route_plan_splits_far_clusters(self) -> None:
        overlay = StubOverlay()
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=120, latitude=120),
            PoiRecord(id="ore_b", title="黄石榴石", category_id=702, longitude=165, latitude=120),
            PoiRecord(id="plant_a", title="向阳花", category_id=705, longitude=760, latitude=760),
            PoiRecord(id="plant_b", title="海桑花", category_id=721, longitude=800, latitude=780),
        ]
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿石"),
            702: PoiCategory(id=702, title="黄石榴石", group_id=1, group_title="矿石"),
            705: PoiCategory(id=705, title="向阳花", group_id=2, group_title="花草"),
            721: PoiCategory(id=721, title="海桑花", group_id=2, group_title="花草"),
        }

        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=1024,
            map_height=1024,
            mode=RESOURCE_ROUTE_MODE_ORE_AND_PLANT,
            start_xy=(100, 100),
            source_label="测试数据",
        )

        self.assertEqual(plan.total_points, 4)
        self.assertEqual(len(plan.segments), 2)
        self.assertEqual(plan.segments[0].points[0].id, "ore_a")

    def test_route_cache_roundtrip(self) -> None:
        overlay = StubOverlay()
        categories = {
            701: PoiCategory(id=701, title="黑晶琉璃", group_id=1, group_title="矿石"),
        }
        records = [
            PoiRecord(id="ore_a", title="黑晶琉璃", category_id=701, longitude=120, latitude=120),
            PoiRecord(id="ore_b", title="黄石榴石", category_id=701, longitude=180, latitude=120),
        ]
        plan = build_resource_route_plan(
            records=records,
            categories=categories,
            overlay=overlay,
            map_width=512,
            map_height=512,
            mode=RESOURCE_ROUTE_MODE_ORE,
            source_label="缓存测试",
        )

        signature = build_route_cache_signature(
            mode=RESOURCE_ROUTE_MODE_ORE,
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
        self.assertEqual(len(restored.segments), len(plan.segments))


if __name__ == "__main__":
    unittest.main()
