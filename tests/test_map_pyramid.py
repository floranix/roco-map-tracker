from __future__ import annotations

import unittest

import numpy as np

from src.map_pyramid import MapPyramid


class MapPyramidTestCase(unittest.TestCase):
    def test_select_level_prefers_nearest_smaller_resize_gap(self) -> None:
        image = np.zeros((4096, 4096, 3), dtype=np.uint8)
        pyramid = MapPyramid(image, min_long_edge=1024)

        level = pyramid.select_level(0.24)

        self.assertAlmostEqual(level.scale, 0.25)

    def test_render_viewport_matches_requested_canvas_size(self) -> None:
        grid_x = np.tile(np.arange(4096, dtype=np.uint16), (4096, 1))
        grid_y = grid_x.T
        image = np.stack(
            [
                (grid_x % 256).astype(np.uint8),
                (grid_y % 256).astype(np.uint8),
                ((grid_x + grid_y) % 256).astype(np.uint8),
            ],
            axis=2,
        )
        pyramid = MapPyramid(image, min_long_edge=1024)

        viewport, level = pyramid.render_viewport(
            target_scale=0.25,
            view_origin=(120, 80),
            canvas_width=320,
            canvas_height=240,
        )

        self.assertEqual(viewport.shape, (240, 320, 3))
        self.assertAlmostEqual(level.scale, 0.25)


if __name__ == "__main__":
    unittest.main()
