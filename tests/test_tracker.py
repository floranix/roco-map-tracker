from __future__ import annotations

import unittest

import numpy as np

from src.tracker import LocalTracker
from src.utils import LocalizationResult


class LocalTrackerTestCase(unittest.TestCase):
    def test_consecutive_failures_force_global_search(self) -> None:
        tracker = LocalTracker(
            roi_expand_pixels=60,
            max_lost_frames=10,
            use_optical_flow=False,
        )
        frame_gray = np.zeros((180, 180), dtype=np.uint8)
        result = LocalizationResult(
            x=320.0,
            y=280.0,
            theta=0.0,
            score=0.9,
            state="tracking",
            method="local_template_match",
            bbox=(230, 190, 410, 370),
        )
        tracker.register_success(frame_gray, result)

        self.assertIsNotNone(tracker.build_search_region(frame_gray, (800, 800)))
        tracker.register_failure()
        self.assertFalse(tracker.should_force_global_search())

        tracker.register_failure()
        self.assertTrue(tracker.should_force_global_search())
        self.assertEqual(tracker.relocalization_aggressiveness(), 1)
        self.assertIsNone(tracker.build_search_region(frame_gray, (800, 800)))

    def test_relocalization_aggressiveness_reaches_second_level(self) -> None:
        tracker = LocalTracker(
            roi_expand_pixels=60,
            max_lost_frames=10,
            use_optical_flow=False,
        )
        tracker.memory.last_result = LocalizationResult(
            x=320.0,
            y=280.0,
            theta=0.0,
            score=0.9,
            state="tracking",
            method="local_template_match",
        )

        for _ in range(5):
            tracker.register_failure()

        self.assertEqual(tracker.relocalization_aggressiveness(), 2)


if __name__ == "__main__":
    unittest.main()
