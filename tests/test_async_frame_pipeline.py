from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from src.async_frame_pipeline import LatestFrameAsyncPipeline


class AsyncFramePipelineTestCase(unittest.TestCase):
    def test_preserves_all_frames_when_processor_keeps_up(self) -> None:
        runner = LatestFrameAsyncPipeline(max_pending_frames=2, poll_interval_seconds=0.01)
        stop_event = threading.Event()
        processed_values: list[int] = []

        def frame_source():
            for index in range(4):
                yield f"frame_{index}", np.full((4, 4, 3), index, dtype=np.uint8)
                time.sleep(0.02)

        def frame_processor(_frame_name: str, frame: np.ndarray) -> None:
            processed_values.append(int(frame[0, 0, 0]))
            time.sleep(0.002)

        stats = runner.run(frame_source(), frame_processor, stop_event)

        self.assertEqual(stats.captured_frames, 4)
        self.assertEqual(stats.processed_frames, 4)
        self.assertEqual(stats.dropped_frames, 0)
        self.assertEqual(processed_values, [0, 1, 2, 3])

    def test_drops_stale_frames_when_capture_outpaces_recognition(self) -> None:
        runner = LatestFrameAsyncPipeline(max_pending_frames=2, poll_interval_seconds=0.01)
        stop_event = threading.Event()
        processed_values: list[int] = []

        def frame_source():
            for index in range(8):
                yield f"frame_{index}", np.full((4, 4, 3), index, dtype=np.uint8)

        def frame_processor(_frame_name: str, frame: np.ndarray) -> None:
            processed_values.append(int(frame[0, 0, 0]))
            time.sleep(0.03)

        stats = runner.run(frame_source(), frame_processor, stop_event)

        self.assertEqual(stats.captured_frames, 8)
        self.assertGreater(stats.dropped_frames, 0)
        self.assertLess(stats.processed_frames, 8)
        self.assertEqual(processed_values, sorted(processed_values))
        self.assertEqual(processed_values[-1], 7)


if __name__ == "__main__":
    unittest.main()
