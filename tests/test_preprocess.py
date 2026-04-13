from __future__ import annotations

import unittest

import numpy as np

from src.preprocess import MapPreprocessor
from src.utils import AppConfig


class PreprocessTestCase(unittest.TestCase):
    def test_minimap_circle_fallback_masks_square_corners(self) -> None:
        frame = np.full((180, 180, 3), 180, dtype=np.uint8)
        config = AppConfig(frame_preprocess_mode="minimap_circle")
        preprocessor = MapPreprocessor(config)

        self.assertIsNone(preprocessor._detect_main_circle(frame))

        prepared = preprocessor.prepare_frame_bundle(frame)

        self.assertIsNotNone(prepared.content_mask)
        assert prepared.content_mask is not None
        self.assertEqual(int(prepared.content_mask[0, 0]), 0)
        self.assertEqual(int(prepared.content_mask[0, -1]), 0)
        self.assertEqual(int(prepared.content_mask[-1, 0]), 0)
        self.assertEqual(int(prepared.content_mask[-1, -1]), 0)
        self.assertGreater(int(prepared.content_mask[60, 90]), 0)

    def test_minimap_circle_fallback_masks_top_center_heading_indicator(self) -> None:
        frame = np.full((180, 180, 3), 180, dtype=np.uint8)
        config = AppConfig(frame_preprocess_mode="minimap_circle")
        preprocessor = MapPreprocessor(config)

        prepared = preprocessor.prepare_frame_bundle(frame)

        self.assertIsNotNone(prepared.content_mask)
        assert prepared.content_mask is not None
        self.assertEqual(int(prepared.content_mask[20, 90]), 0)
        self.assertGreater(int(prepared.content_mask[60, 90]), 0)


if __name__ == "__main__":
    unittest.main()
