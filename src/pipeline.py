from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from src.feature_matcher import FeatureMatcher
from src.global_localizer import GlobalLocalizer
from src.kalman_filter import PositionKalmanFilter
from src.preprocess import MapPreprocessor
from src.tracker import LocalTracker
from src.utils import AppConfig, LocalizationResult, offset_result_geometry


@dataclass(frozen=True)
class RelocalizationOverrides:
    frame_scales: list[float]
    template_scales: list[float]
    template_top_per_scale: int
    template_top_k: int
    template_refine_radius: int
    template_min_score: float
    global_tile_top_k: int


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
        )
        self.tracker = LocalTracker(
            roi_expand_pixels=config.roi_expand_pixels,
            max_lost_frames=config.max_lost_frames,
            use_optical_flow=config.use_optical_flow,
            motion_gate_pixels=config.tracking_motion_gate_pixels,
            motion_gate_per_lost_frame=config.tracking_motion_gate_per_lost_frame,
        )
        self.kalman = PositionKalmanFilter() if config.use_kalman else None

    def process_frame(self, frame: np.ndarray) -> LocalizationResult:
        prepared_frame = self.preprocessor.prepare_frame_bundle(frame)
        frame_gray = prepared_frame.gray
        search_region = self.tracker.build_search_region(frame_gray, self.map_bundle.gray.shape)

        result = None
        if search_region is not None:
            fast_template_result = self.localizer.localize_template(
                frame_gray=frame_gray,
                content_mask=prepared_frame.content_mask,
                search_region=search_region,
                state="tracking",
                template_scales=self.config.tracking_template_scales,
                top_per_scale=self.config.tracking_template_top_per_scale,
                top_k=self.config.tracking_template_top_k,
                refine_radius=self.config.tracking_template_refine_radius,
                min_score=self.config.tracking_template_min_score,
            )
            if (
                fast_template_result is not None
                and fast_template_result.score >= self.config.tracking_template_early_accept_score
                and self.tracker.is_result_plausible(fast_template_result, frame_gray.shape, strict=True)
            ):
                result = fast_template_result

        if result is None and search_region is not None:
            result = self.localizer.localize(
                frame_gray,
                frame_mask=prepared_frame.feature_mask,
                content_mask=prepared_frame.content_mask,
                search_region=search_region,
                state="tracking",
            )
            if result is not None and not self.tracker.is_result_plausible(result, frame_gray.shape, strict=True):
                result = None

        if result is None:
            relocalization_overrides = self._build_relocalization_overrides()
            result = self.localizer.localize(
                frame_gray,
                frame_mask=prepared_frame.feature_mask,
                content_mask=prepared_frame.content_mask,
                search_region=None,
                state="relocalizing",
                frame_scales=relocalization_overrides.frame_scales,
                template_scales=relocalization_overrides.template_scales,
                template_top_per_scale=relocalization_overrides.template_top_per_scale,
                template_top_k=relocalization_overrides.template_top_k,
                template_refine_radius=relocalization_overrides.template_refine_radius,
                template_min_score=relocalization_overrides.template_min_score,
                global_tile_top_k=relocalization_overrides.global_tile_top_k,
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

    def _build_relocalization_overrides(self) -> RelocalizationOverrides:
        aggressiveness = self.tracker.relocalization_aggressiveness()
        frame_scales = list(self.config.global_search_scales)
        template_scales = list(self.config.template_match_scales)
        template_top_per_scale = int(self.config.template_match_top_per_scale)
        template_top_k = int(self.config.template_match_top_k)
        template_refine_radius = int(self.config.template_match_refine_radius)
        template_min_score = float(self.config.template_match_min_score)
        global_tile_top_k = int(self.config.global_tile_top_k)

        if aggressiveness >= 1:
            frame_scales = self._merge_scales(frame_scales, [0.2, 0.3, 1.8, 2.2])
            template_scales = self._merge_scales(template_scales, [0.55, 0.6, 1.65, 1.8])
            template_top_per_scale += 1
            template_top_k += 3
            template_refine_radius = int(round(template_refine_radius * 1.5))
            template_min_score = max(0.58, template_min_score - 0.08)
            global_tile_top_k += 4

        if aggressiveness >= 2:
            frame_scales = self._merge_scales(frame_scales, [0.18, 2.5])
            template_scales = self._merge_scales(template_scales, [0.5, 1.95])
            template_top_per_scale += 1
            template_top_k += 2
            template_refine_radius = int(round(template_refine_radius * 1.35))
            template_min_score = max(0.52, template_min_score - 0.06)
            global_tile_top_k += 4

        return RelocalizationOverrides(
            frame_scales=frame_scales,
            template_scales=template_scales,
            template_top_per_scale=template_top_per_scale,
            template_top_k=template_top_k,
            template_refine_radius=template_refine_radius,
            template_min_score=template_min_score,
            global_tile_top_k=global_tile_top_k,
        )

    @staticmethod
    def _merge_scales(base_scales: list[float], extra_scales: list[float]) -> list[float]:
        merged = []
        for raw_scale in [*base_scales, *extra_scales]:
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                continue
            if scale <= 0:
                continue
            if not any(abs(scale - existing) < 1e-6 for existing in merged):
                merged.append(scale)
        return merged
