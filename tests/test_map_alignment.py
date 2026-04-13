from __future__ import annotations

import unittest

from src.map_alignment import resolve_map_alignment
from src.utils import LocalizationResult


class MapAlignmentTests(unittest.TestCase):
    def test_resolve_known_alignment_between_17173_and_biliwiki(self) -> None:
        alignment = resolve_map_alignment(
            "data/rocom_17173/rocom-shijie-z13.png",
            "data/rocom_biliwiki/rocom_base_z8.png",
        )

        self.assertIsNotNone(alignment)
        projected = alignment.project_point(5417.594, 4462.594)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], 8888.655, places=1)
        self.assertAlmostEqual(projected[1], 4475.400, places=1)

    def test_result_projection_keeps_geometry_consistent(self) -> None:
        alignment = resolve_map_alignment(
            "data/rocom_17173/rocom-shijie-z13.png",
            "data/rocom_biliwiki/rocom_base_z8.png",
        )
        self.assertIsNotNone(alignment)

        result = LocalizationResult(
            x=5417.594,
            y=4462.594,
            theta=0.0,
            score=0.84,
            state="relocalizing",
            method="global_template_match",
            bbox=(5308, 4353, 5528, 4573),
            corners=[
                (5308.0, 4353.0),
                (5527.18798828125, 4353.0),
                (5527.18798828125, 4572.18798828125),
                (5308.0, 4572.18798828125),
            ],
        )

        projected = alignment.project_result(result)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected.x, 8888.655, places=1)
        self.assertAlmostEqual(projected.y, 4475.400, places=1)
        self.assertEqual(len(projected.corners), 4)
        self.assertLess(projected.bbox[0], projected.bbox[2])
        self.assertLess(projected.bbox[1], projected.bbox[3])

    def test_same_coordinate_space_uses_identity_alignment(self) -> None:
        alignment = resolve_map_alignment(
            "data/rocom_biliwiki/rocom_base_z8.png",
            "data/rocom_biliwiki/rocom_caiji_overlay.png",
        )

        self.assertIsNotNone(alignment)
        projected = alignment.project_point(9003.868, 4587.2764)
        self.assertIsNotNone(projected)
        self.assertAlmostEqual(projected[0], 9003.868, places=3)
        self.assertAlmostEqual(projected[1], 4587.2764, places=3)


if __name__ == "__main__":
    unittest.main()
