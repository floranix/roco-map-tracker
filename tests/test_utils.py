from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.utils import draw_position_marker, guess_poi_categories_path, guess_poi_icon_dir


class UtilsTestCase(unittest.TestCase):
    def test_guess_poi_categories_path_returns_empty_for_blank_input(self) -> None:
        self.assertEqual(guess_poi_categories_path(""), "")

    def test_guess_poi_icon_dir_returns_empty_for_blank_input(self) -> None:
        self.assertEqual(guess_poi_icon_dir(""), "")

    def test_draw_position_marker_uses_icon_when_available(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            icon_path = Path(temp_dir) / "marker.png"
            icon = np.zeros((20, 20, 4), dtype=np.uint8)
            icon[:, :, 2] = 255
            icon[:, :, 3] = 255
            self.assertTrue(cv2.imwrite(str(icon_path), icon))

            canvas = np.zeros((60, 60, 3), dtype=np.uint8)
            with patch("src.utils.POSITION_MARKER_ICON_PATH", icon_path):
                draw_position_marker(canvas, (30, 30), target_edge=24, icon_path=icon_path)

        self.assertGreater(int(canvas[:, :, 2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
