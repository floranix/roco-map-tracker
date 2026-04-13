from __future__ import annotations

import math

import numpy as np

from src.feature_matcher import FeatureMatcher
from src.global_localizer import GlobalLocalizer
from src.kalman_filter import PositionKalmanFilter
from src.preprocess import MapPreprocessor
from src.tracker import LocalTracker
from src.utils import AppConfig, LocalizationResult, offset_result_geometry


class LocalizationPipeline:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.preprocessor = MapPreprocessor(config)
        self.map_bundle = self.preprocessor.load_map()
        self.matcher = FeatureMatcher(config)
        self.localizer = GlobalLocalizer(
            map_gray=self.map_bundle.gray,
            matcher=self.matcher,
            min_match_count=config.min_match_count,
            ransac_threshold=config.ransac_threshold,
            frame_scales=config.global_search_scales,
            global_tile_size=config.global_tile_size,
            global_tile_stride=config.global_tile_stride,
            global_tile_top_k=config.global_tile_top_k,
            max_rotation_degrees=config.max_rotation_degrees,
            candidate_verification_weight=config.candidate_verification_weight,
            use_template_matching=config.use_template_matching,
            template_match_map_downsample=config.template_match_map_downsample,
            template_match_scales=config.template_match_scales,
            template_match_top_per_scale=config.template_match_top_per_scale,
            template_match_top_k=config.template_match_top_k,
            template_match_refine_radius=config.template_match_refine_radius,
            template_match_min_score=config.template_match_min_score,
            template_match_blur_size=config.template_match_blur_size,
            template_color_validation_weight=config.template_color_validation_weight,
            template_color_validation_min_score=config.template_color_validation_min_score,
        )
        self.tracker = LocalTracker(
            roi_expand_pixels=config.roi_expand_pixels,
            max_lost_frames=config.max_lost_frames,
            use_optical_flow=config.use_optical_flow,
        )
        self.kalman = PositionKalmanFilter() if config.use_kalman else None

    def process_frame(self, frame: np.ndarray) -> LocalizationResult:
        prepared_frame = self.preprocessor.prepare_frame_bundle(frame)
        frame_gray = prepared_frame.gray
        search_region = self.tracker.build_search_region(frame_gray, self.map_bundle.gray.shape)

        result = None
        if search_region is not None:
            result = self.localizer.localize(
                frame_gray,
                frame_mask=prepared_frame.feature_mask,
                content_mask=prepared_frame.content_mask,
                search_region=search_region,
                state="tracking",
            )

        if result is None:
            result = self.localizer.localize(
                frame_gray,
                frame_mask=prepared_frame.feature_mask,
                content_mask=prepared_frame.content_mask,
                search_region=None,
                state="relocalizing",
            )

        if result is not None:
            result = self._apply_smoothing(result)
            self.tracker.register_success(frame_gray, result)
            return result

        self.tracker.register_failure()
        return self._fallback_result()

    def _apply_smoothing(self, result: LocalizationResult) -> LocalizationResult:
        if self.kalman is None:
            return result

        if self.kalman.initialized:
            self.kalman.predict()
        smoothed_x, smoothed_y = self.kalman.correct(result.x, result.y)
        dx = smoothed_x - result.x
        dy = smoothed_y - result.y
        result.x = smoothed_x
        result.y = smoothed_y
        return offset_result_geometry(result, dx, dy)

    def _fallback_result(self) -> LocalizationResult:
        predicted = self.kalman.predict() if self.kalman is not None else None
        if predicted is not None:
            x, y = predicted
        elif self.tracker.memory.last_result is not None:
            x = self.tracker.memory.last_result.x
            y = self.tracker.memory.last_result.y
        else:
            x = math.nan
            y = math.nan

        state = "lost" if self.tracker.is_lost() else "relocalizing"
        score = 0.0 if state == "lost" else 0.1
        return LocalizationResult(
            x=x,
            y=y,
            theta=None,
            score=score,
            state=state,
            method="prediction",
        )
