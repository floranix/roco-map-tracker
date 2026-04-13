from __future__ import annotations

import unittest

import numpy as np

from src.global_localizer import GlobalLocalizer


class GlobalLocalizerTestCase(unittest.TestCase):
    def test_masked_color_similarity_prefers_closer_hue(self) -> None:
        mask = np.full((32, 32), 255, dtype=np.uint8)
        frame = np.full((32, 32, 3), (90, 170, 220), dtype=np.uint8)
        similar = np.full((32, 32, 3), (96, 176, 222), dtype=np.uint8)
        different = np.full((32, 32, 3), (96, 176, 120), dtype=np.uint8)

        similar_score = GlobalLocalizer._masked_color_similarity(frame, similar, mask)
        different_score = GlobalLocalizer._masked_color_similarity(frame, different, mask)

        self.assertGreater(similar_score, different_score)
        self.assertGreater(similar_score, 0.85)
        self.assertLess(different_score, 0.7)

    def test_small_global_frame_expands_template_candidate_limits(self) -> None:
        top_per_scale, top_k = GlobalLocalizer._adjust_template_candidate_limits(
            frame_shape=(125, 122),
            search_region=None,
            top_per_scale=2,
            top_k=6,
        )

        self.assertEqual(top_per_scale, 4)
        self.assertEqual(top_k, 10)

    def test_large_or_local_frame_keeps_template_candidate_limits(self) -> None:
        large_top_per_scale, large_top_k = GlobalLocalizer._adjust_template_candidate_limits(
            frame_shape=(180, 180),
            search_region=None,
            top_per_scale=2,
            top_k=6,
        )
        local_top_per_scale, local_top_k = GlobalLocalizer._adjust_template_candidate_limits(
            frame_shape=(125, 122),
            search_region=(0, 0, 256, 256),
            top_per_scale=2,
            top_k=6,
        )

        self.assertEqual((large_top_per_scale, large_top_k), (2, 6))
        self.assertEqual((local_top_per_scale, local_top_k), (2, 6))


if __name__ == "__main__":
    unittest.main()
