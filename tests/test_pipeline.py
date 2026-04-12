from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.pipeline import LocalizationPipeline
from src.utils import AppConfig


def build_synthetic_map(size: int = 720) -> np.ndarray:
    canvas = np.full((size, size, 3), 245, dtype=np.uint8)

    for index in range(40, size, 80):
        cv2.line(canvas, (index, 0), (index, size - 1), (40, 40, 40), 2)
        cv2.line(canvas, (0, index), (size - 1, index), (40, 40, 40), 2)

    for i, center in enumerate([(120, 150), (280, 330), (510, 220), (610, 580), (180, 560), (430, 470)]):
        cv2.circle(canvas, center, 26 + i * 3, (20 + i * 20, 90, 180), -1)
        cv2.putText(
            canvas,
            f"P{i}",
            (center[0] - 18, center[1] + 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(canvas, (320, 70), (520, 170), (30, 170, 40), -1)
    cv2.rectangle(canvas, (80, 360), (220, 520), (160, 60, 60), -1)
    cv2.putText(canvas, "ROCO", (310, 130), cv2.FONT_HERSHEY_DUPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, "MAP", (100, 455), cv2.FONT_HERSHEY_DUPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)

    polygon = np.array([[540, 320], [660, 380], [610, 470], [500, 430]], dtype=np.int32)
    cv2.fillPoly(canvas, [polygon], (60, 100, 220))
    cv2.polylines(canvas, [polygon], True, (255, 255, 255), 3)

    noise = np.random.default_rng(7).integers(0, 20, size=canvas.shape, dtype=np.uint8)
    return cv2.add(canvas, noise)


def crop_frame(image: np.ndarray, center_x: int, center_y: int, width: int = 180, height: int = 180) -> np.ndarray:
    x0 = center_x - width // 2
    y0 = center_y - height // 2
    return image[y0 : y0 + height, x0 : x0 + width].copy()


class PipelineTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.map_path = Path(self.temp_dir.name) / "full_map.png"
        synthetic_map = build_synthetic_map()
        cv2.imwrite(str(self.map_path), synthetic_map)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_single_frame_global_localization(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=80,
                use_optical_flow=False,
                use_kalman=False,
            )
        )

        frame = crop_frame(cv2.imread(str(self.map_path)), 370, 315)
        result = pipeline.process_frame(frame)

        self.assertEqual(result.state, "relocalizing")
        self.assertLess(abs(result.x - 370), 15)
        self.assertLess(abs(result.y - 315), 15)
        self.assertGreater(result.score, 0.4)

    def test_multi_frame_tracking(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=60,
                use_optical_flow=True,
                use_kalman=True,
            )
        )

        centers = [(330, 300), (344, 312), (360, 326), (378, 340)]
        results = []
        source = cv2.imread(str(self.map_path))
        for center_x, center_y in centers:
            frame = crop_frame(source, center_x, center_y)
            results.append(pipeline.process_frame(frame))

        self.assertEqual(results[0].state, "relocalizing")
        self.assertIn("tracking", {result.state for result in results[1:]})
        self.assertLess(abs(results[-1].x - centers[-1][0]), 20)
        self.assertLess(abs(results[-1].y - centers[-1][1]), 20)
        self.assertGreater(results[-1].score, 0.3)

    def test_global_localization_with_zoomed_frame(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=80,
                use_optical_flow=False,
                use_kalman=False,
                global_search_scales=[1.0, 0.85, 0.7, 0.55, 1.25, 1.5],
            )
        )

        source = cv2.imread(str(self.map_path))
        zoomed_crop = crop_frame(source, 430, 470, width=120, height=120)
        zoomed_frame = cv2.resize(zoomed_crop, (180, 180), interpolation=cv2.INTER_LINEAR)

        result = pipeline.process_frame(zoomed_frame)

        self.assertNotEqual(result.state, "lost")
        self.assertLess(abs(result.x - 430), 20)
        self.assertLess(abs(result.y - 470), 20)
        self.assertGreater(result.score, 0.25)


if __name__ == "__main__":
    unittest.main()
