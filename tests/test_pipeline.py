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


def build_minimap_like_frame(frame: np.ndarray) -> np.ndarray:
    minimap = frame.copy()
    height, width = minimap.shape[:2]
    center = (width // 2, height // 2)

    cv2.circle(minimap, center, 16, (255, 255, 255), -1)
    arrow = np.array(
        [
            [center[0] - 20, center[1] + 6],
            [center[0] + 2, center[1] - 12],
            [center[0] + 16, center[1] + 4],
            [center[0] - 2, center[1] + 22],
        ],
        dtype=np.int32,
    )
    cv2.fillConvexPoly(minimap, arrow, (40, 150, 255))
    cv2.polylines(minimap, [arrow], True, (255, 255, 255), 3)

    target_marker = np.array([[74, 18], [92, 0], [110, 18], [92, 36]], dtype=np.int32)
    cv2.fillConvexPoly(minimap, target_marker, (210, 120, 255))
    cv2.polylines(minimap, [target_marker], True, (255, 255, 255), 2)
    cv2.circle(minimap, (92, 18), 6, (235, 235, 235), -1)

    cv2.circle(minimap, (138, 142), 16, (255, 255, 255), -1)
    cv2.circle(minimap, (138, 142), 13, (215, 120, 50), -1)
    cv2.putText(
        minimap,
        "2",
        (132, 149),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return minimap


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

    def test_repeated_frame_uses_fast_local_template_tracking(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=60,
                use_optical_flow=True,
                use_kalman=False,
                frame_preprocess_mode="minimap_circle",
            )
        )

        source = cv2.imread(str(self.map_path))
        frame = build_minimap_like_frame(crop_frame(source, 430, 470))

        first = pipeline.process_frame(frame)
        second = pipeline.process_frame(frame)

        self.assertEqual(first.state, "relocalizing")
        self.assertEqual(second.state, "tracking")
        self.assertEqual(second.method, "local_template_match")
        self.assertLess(abs(second.x - 430), 24)
        self.assertLess(abs(second.y - 470), 24)
        self.assertGreater(second.score, 0.3)

    def test_large_jump_triggers_relocalization_instead_of_bad_local_lock(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=60,
                use_optical_flow=True,
                use_kalman=False,
            )
        )

        source = cv2.imread(str(self.map_path))
        first = pipeline.process_frame(crop_frame(source, 330, 300))
        second = pipeline.process_frame(crop_frame(source, 610, 580))

        self.assertEqual(first.state, "relocalizing")
        self.assertNotEqual(second.state, "lost")
        self.assertLess(abs(second.x - 610), 24)
        self.assertLess(abs(second.y - 580), 24)
        self.assertIn(second.method, {"global_template_match", "global_feature_match", "global_tile_match"})

    def test_consecutive_failures_escalate_to_global_relocalization(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=60,
                use_optical_flow=True,
                use_kalman=False,
            )
        )

        source = cv2.imread(str(self.map_path))
        first = pipeline.process_frame(crop_frame(source, 330, 300))
        blank = np.zeros((180, 180, 3), dtype=np.uint8)
        pipeline.process_frame(blank)
        pipeline.process_frame(blank)
        second = pipeline.process_frame(crop_frame(source, 610, 580))

        self.assertEqual(first.state, "relocalizing")
        self.assertFalse(np.isnan(second.x))
        self.assertFalse(np.isnan(second.y))
        self.assertLess(abs(second.x - 610), 24)
        self.assertLess(abs(second.y - 580), 24)
        self.assertIn(second.method, {"global_template_match", "global_feature_match", "global_tile_match"})

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

    def test_minimap_overlay_frame_avoids_degenerate_false_positive(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=80,
                use_optical_flow=False,
                use_kalman=False,
                frame_preprocess_mode="minimap_circle",
                global_search_scales=[1.0, 0.85, 0.7, 0.55, 1.2, 1.4],
            )
        )

        source = cv2.imread(str(self.map_path))
        frame = crop_frame(source, 430, 470)
        minimap_frame = build_minimap_like_frame(frame)

        result = pipeline.process_frame(minimap_frame)

        self.assertNotEqual(result.state, "lost")
        self.assertFalse(np.isnan(result.x))
        self.assertFalse(np.isnan(result.y))
        self.assertLess(abs(result.x - 430), 28)
        self.assertLess(abs(result.y - 470), 28)
        self.assertGreater(result.score, 0.2)

    def test_rotated_frame_is_rejected_when_rotation_is_disabled(self) -> None:
        pipeline = LocalizationPipeline(
            AppConfig(
                map_path=str(self.map_path),
                feature_type="orb",
                min_match_count=8,
                roi_expand_pixels=80,
                use_optical_flow=False,
                use_kalman=False,
                global_search_scales=[1.0, 0.85, 0.7, 0.55, 1.25, 1.5],
                max_rotation_degrees=8.0,
            )
        )

        source = cv2.imread(str(self.map_path))
        frame = crop_frame(source, 430, 470, width=180, height=180)
        rotation_matrix = cv2.getRotationMatrix2D((90, 90), 25, 1.0)
        rotated_frame = cv2.warpAffine(
            frame,
            rotation_matrix,
            (180, 180),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        result = pipeline.process_frame(rotated_frame)

        self.assertEqual(result.state, "relocalizing")
        self.assertTrue(np.isnan(result.x))
        self.assertTrue(np.isnan(result.y))


if __name__ == "__main__":
    unittest.main()
