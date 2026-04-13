from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import cv2
import numpy as np

from src.feature_matcher import FeatureMatcher
from src.utils import LocalizationResult


@dataclass
class GlobalTile:
    x0: int
    y0: int
    x1: int
    y1: int
    keypoints: list
    descriptors: np.ndarray | None


@dataclass
class TemplateMatchCandidate:
    x0: int
    y0: int
    width: int
    height: int
    scale: float
    score: float


class GlobalLocalizer:
    def __init__(
        self,
        map_gray: np.ndarray,
        matcher: FeatureMatcher,
        min_match_count: int,
        ransac_threshold: float,
        frame_scales: list[float] | None = None,
        global_tile_size: int = 1024,
        global_tile_stride: int = 768,
        global_tile_top_k: int = 8,
        max_rotation_degrees: float = 8.0,
        candidate_verification_weight: float = 0.55,
        use_template_matching: bool = True,
        template_match_map_downsample: float = 0.25,
        template_match_scales: list[float] | None = None,
        template_match_top_per_scale: int = 2,
        template_match_top_k: int = 6,
        template_match_refine_radius: int = 420,
        template_match_min_score: float = 0.72,
        template_match_blur_size: int = 7,
    ) -> None:
        self.map_gray = map_gray
        self.matcher = matcher
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold
        self.frame_scales = self._normalize_scales(frame_scales or [1.0])
        self.global_tile_size = global_tile_size
        self.global_tile_stride = global_tile_stride
        self.global_tile_top_k = global_tile_top_k
        self.max_rotation_degrees = max(0.0, float(max_rotation_degrees))
        self.candidate_verification_weight = float(np.clip(candidate_verification_weight, 0.0, 1.0))
        self.use_template_matching = bool(use_template_matching)
        self.template_match_map_downsample = float(np.clip(template_match_map_downsample, 0.05, 1.0))
        self.template_match_scales = self._normalize_scales(template_match_scales or [1.0])
        self.template_match_top_per_scale = max(1, int(template_match_top_per_scale))
        self.template_match_top_k = max(1, int(template_match_top_k))
        self.template_match_refine_radius = max(64, int(template_match_refine_radius))
        self.template_match_min_score = float(np.clip(template_match_min_score, 0.0, 1.0))
        self.template_match_blur_size = self._normalize_blur_size(template_match_blur_size)
        self.map_keypoints, self.map_descriptors = matcher.detect_and_compute(map_gray)
        self.global_tiles = self._build_global_tiles()
        self.map_gray_blurred = cv2.GaussianBlur(
            self.map_gray,
            (self.template_match_blur_size, self.template_match_blur_size),
            0,
        )
        self.template_map_gray = self._resize_frame(self.map_gray, self.template_match_map_downsample)
        if self.template_map_gray is None:
            self.template_map_gray = self.map_gray.copy()
            self.template_match_map_downsample = 1.0
        self.template_map_gray_blurred = cv2.GaussianBlur(
            self.template_map_gray,
            (self.template_match_blur_size, self.template_match_blur_size),
            0,
        )

    def localize(
        self,
        frame_gray: np.ndarray,
        frame_mask: np.ndarray | None = None,
        content_mask: np.ndarray | None = None,
        search_region: Optional[tuple[int, int, int, int]] = None,
        state: str = "relocalizing",
    ) -> Optional[LocalizationResult]:
        if search_region is None:
            map_keypoints = self.map_keypoints
            map_descriptors = self.map_descriptors
            method = "global_feature_match"
        else:
            x0, y0, x1, y1 = search_region
            roi = self.map_gray[y0:y1, x0:x1]
            map_keypoints, map_descriptors = self.matcher.detect_and_compute(roi)
            map_keypoints = self.matcher.shift_keypoints(map_keypoints, x0, y0)
            method = "local_feature_match"

        best_result: Optional[LocalizationResult] = None
        if content_mask is not None and self.use_template_matching:
            template_result = self._localize_with_template_matching(
                frame_gray=frame_gray,
                content_mask=content_mask,
                search_region=search_region,
                state=state,
            )
            best_result = self._update_best_result(best_result, template_result)

        for scale in self.frame_scales:
            scaled_frame = self._resize_frame(frame_gray, scale)
            if scaled_frame is None:
                continue
            scaled_mask = self._resize_mask(frame_mask, scale)

            frame_keypoints, frame_descriptors = self.matcher.detect_and_compute(scaled_frame, mask=scaled_mask)
            if len(frame_keypoints) < 4 or frame_descriptors is None:
                continue

            if search_region is None:
                tile_result = self._search_global_tiles(
                    frame_gray=frame_gray,
                    content_mask=content_mask,
                    frame_keypoints=frame_keypoints,
                    frame_descriptors=frame_descriptors,
                    scale=scale,
                    state=state,
                )
                best_result = self._update_best_result(best_result, tile_result)

            direct_result = self._localize_from_descriptors(
                frame_gray=frame_gray,
                content_mask=content_mask,
                frame_keypoints=frame_keypoints,
                frame_descriptors=frame_descriptors,
                map_keypoints=map_keypoints,
                map_descriptors=map_descriptors,
                scale=scale,
                state=state,
                method=method,
            )
            best_result = self._update_best_result(best_result, direct_result)

        return best_result

    def _search_global_tiles(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray | None,
        frame_keypoints,
        frame_descriptors,
        scale: float,
        state: str,
    ) -> Optional[LocalizationResult]:
        if not self.global_tiles or self.global_tile_top_k <= 0:
            return None

        ranked_tiles = []
        for tile in self.global_tiles:
            matches = self.matcher.match(frame_descriptors, tile.descriptors)
            if len(matches) < self.min_match_count:
                continue
            ranked_tiles.append((len(matches), matches, tile))

        ranked_tiles.sort(key=lambda item: item[0], reverse=True)
        best_result: Optional[LocalizationResult] = None
        for _match_count, matches, tile in ranked_tiles[: self.global_tile_top_k]:
            result = self._localize_from_candidate(
                frame_gray=frame_gray,
                content_mask=content_mask,
                frame_keypoints=frame_keypoints,
                map_keypoints=tile.keypoints,
                matches=matches,
                scale=scale,
                state=state,
                method="global_tile_match",
            )
            best_result = self._update_best_result(best_result, result)
        return best_result

    def _localize_from_descriptors(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray | None,
        frame_keypoints,
        frame_descriptors,
        map_keypoints,
        map_descriptors,
        scale: float,
        state: str,
        method: str,
    ) -> Optional[LocalizationResult]:
        matches = self.matcher.match(frame_descriptors, map_descriptors)
        if len(matches) < self.min_match_count:
            return None
        return self._localize_from_candidate(
            frame_gray=frame_gray,
            content_mask=content_mask,
            frame_keypoints=frame_keypoints,
            map_keypoints=map_keypoints,
            matches=matches,
            scale=scale,
            state=state,
            method=method,
        )

    def _localize_from_candidate(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray | None,
        frame_keypoints,
        map_keypoints,
        matches,
        scale: float,
        state: str,
        method: str,
    ) -> Optional[LocalizationResult]:
        homography, inlier_mask = self._estimate_transform(frame_keypoints, map_keypoints, matches)
        if homography is None or inlier_mask is None:
            return None

        inliers = int(inlier_mask.ravel().sum())
        if inliers < self.min_match_count:
            return None

        adjusted_homography = self._adjust_homography_for_scale(homography, scale)
        feature_score = self._score_matches(len(matches), inliers)
        verification_score = self._verify_candidate(frame_gray, content_mask, adjusted_homography)
        if verification_score is not None and verification_score < 0.12:
            return None
        return self._try_build_result(
            frame_gray=frame_gray,
            homography=adjusted_homography,
            match_count=len(matches),
            inlier_count=inliers,
            feature_score=feature_score,
            verification_score=verification_score,
            state=state,
            method=method,
        )

    def _estimate_transform(self, frame_keypoints, map_keypoints, matches):
        source_points = np.float32(
            [frame_keypoints[match.queryIdx].pt for match in matches]
        ).reshape(-1, 1, 2)
        destination_points = np.float32(
            [map_keypoints[match.trainIdx].pt for match in matches]
        ).reshape(-1, 1, 2)

        homography, inlier_mask = cv2.findHomography(
            source_points,
            destination_points,
            cv2.RANSAC,
            self.ransac_threshold,
        )
        if homography is not None and inlier_mask is not None:
            return homography, inlier_mask

        affine, affine_mask = cv2.estimateAffinePartial2D(
            source_points,
            destination_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=self.ransac_threshold,
        )
        if affine is None or affine_mask is None:
            return None, None

        homography = np.vstack([affine, np.array([0.0, 0.0, 1.0])])
        return homography, affine_mask

    def _localize_with_template_matching(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray,
        search_region: Optional[tuple[int, int, int, int]],
        state: str,
    ) -> Optional[LocalizationResult]:
        coarse_candidates: list[TemplateMatchCandidate]
        if search_region is None:
            coarse_candidates = self._search_template_candidates(
                map_gray=self.template_map_gray,
                map_gray_blurred=self.template_map_gray_blurred,
                map_offset_x=0,
                map_offset_y=0,
                map_scale=self.template_match_map_downsample,
                frame_gray=frame_gray,
                content_mask=content_mask,
                template_scales=self.template_match_scales,
                top_per_scale=self.template_match_top_per_scale,
            )
            refined_candidates = self._refine_template_candidates(
                frame_gray=frame_gray,
                content_mask=content_mask,
                coarse_candidates=coarse_candidates,
            )
            method = "global_template_match"
        else:
            x0, y0, x1, y1 = search_region
            roi_blurred = self.map_gray_blurred[y0:y1, x0:x1]
            refined_candidates = self._search_template_candidates(
                map_gray=self.map_gray[y0:y1, x0:x1],
                map_gray_blurred=roi_blurred,
                map_offset_x=x0,
                map_offset_y=y0,
                map_scale=1.0,
                frame_gray=frame_gray,
                content_mask=content_mask,
                template_scales=self.template_match_scales,
                top_per_scale=self.template_match_top_per_scale,
            )
            method = "local_template_match"

        best_result: Optional[LocalizationResult] = None
        for candidate in refined_candidates[: self.template_match_top_k]:
            homography = self._template_candidate_homography(candidate)
            verification_score = self._verify_candidate(frame_gray, content_mask, homography)
            if verification_score is not None and verification_score < self.template_match_min_score * 0.75:
                continue
            result = self._try_build_result(
                frame_gray=frame_gray,
                homography=homography,
                match_count=0,
                inlier_count=0,
                feature_score=candidate.score,
                verification_score=verification_score,
                state=state,
                method=method,
            )
            best_result = self._update_best_result(best_result, result)

        if best_result is None:
            return None
        if best_result.score < self.template_match_min_score:
            return None
        return best_result

    def _refine_template_candidates(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray,
        coarse_candidates: list[TemplateMatchCandidate],
    ) -> list[TemplateMatchCandidate]:
        refined: list[TemplateMatchCandidate] = []
        height, width = self.map_gray.shape[:2]
        for coarse_candidate in coarse_candidates[: self.template_match_top_k]:
            center_x = coarse_candidate.x0 + coarse_candidate.width // 2
            center_y = coarse_candidate.y0 + coarse_candidate.height // 2
            half_window = max(
                self.template_match_refine_radius,
                int(round(max(coarse_candidate.width, coarse_candidate.height) * 1.25)),
            )
            x0 = max(0, center_x - half_window)
            y0 = max(0, center_y - half_window)
            x1 = min(width, center_x + half_window)
            y1 = min(height, center_y + half_window)
            if x1 - x0 <= frame_gray.shape[1] or y1 - y0 <= frame_gray.shape[0]:
                continue

            roi_blurred = self.map_gray_blurred[y0:y1, x0:x1]
            refine_scales = self._normalize_scales(
                [
                    coarse_candidate.scale * 0.9,
                    coarse_candidate.scale * 0.95,
                    coarse_candidate.scale,
                    coarse_candidate.scale * 1.05,
                    coarse_candidate.scale * 1.1,
                ]
            )
            refined.extend(
                self._search_template_candidates(
                    map_gray=self.map_gray[y0:y1, x0:x1],
                    map_gray_blurred=roi_blurred,
                    map_offset_x=x0,
                    map_offset_y=y0,
                    map_scale=1.0,
                    frame_gray=frame_gray,
                    content_mask=content_mask,
                    template_scales=refine_scales,
                    top_per_scale=1,
                )
            )

        refined.sort(key=lambda candidate: candidate.score, reverse=True)
        return self._deduplicate_template_candidates(refined)

    def _search_template_candidates(
        self,
        map_gray: np.ndarray,
        map_gray_blurred: np.ndarray,
        map_offset_x: int,
        map_offset_y: int,
        map_scale: float,
        frame_gray: np.ndarray,
        content_mask: np.ndarray,
        template_scales: list[float],
        top_per_scale: int,
    ) -> list[TemplateMatchCandidate]:
        candidates: list[TemplateMatchCandidate] = []
        if map_gray.shape[0] < 32 or map_gray.shape[1] < 32:
            return candidates

        for scale in template_scales:
            template_width = max(24, int(round(frame_gray.shape[1] * scale * map_scale)))
            template_height = max(24, int(round(frame_gray.shape[0] * scale * map_scale)))
            if template_width >= map_gray.shape[1] or template_height >= map_gray.shape[0]:
                continue

            template = cv2.resize(
                frame_gray,
                (template_width, template_height),
                interpolation=cv2.INTER_AREA if template_width < frame_gray.shape[1] else cv2.INTER_LINEAR,
            )
            template_mask = cv2.resize(
                content_mask,
                (template_width, template_height),
                interpolation=cv2.INTER_NEAREST,
            )
            if np.count_nonzero(template_mask) < max(400, template_width * template_height * 0.15):
                continue

            template_blurred = cv2.GaussianBlur(
                template,
                (self.template_match_blur_size, self.template_match_blur_size),
                0,
            )
            try:
                response = cv2.matchTemplate(
                    map_gray_blurred,
                    template_blurred,
                    cv2.TM_CCORR_NORMED,
                    mask=template_mask,
                )
            except cv2.error:
                continue

            candidates.extend(
                self._extract_template_candidates(
                    response=response,
                    map_offset_x=map_offset_x,
                    map_offset_y=map_offset_y,
                    map_scale=map_scale,
                    frame_width=frame_gray.shape[1],
                    frame_height=frame_gray.shape[0],
                    template_width=template_width,
                    template_height=template_height,
                    top_k=top_per_scale,
                )
            )

        candidates.sort(key=lambda candidate: candidate.score, reverse=True)
        return self._deduplicate_template_candidates(candidates)

    def _extract_template_candidates(
        self,
        response: np.ndarray,
        map_offset_x: int,
        map_offset_y: int,
        map_scale: float,
        frame_width: int,
        frame_height: int,
        template_width: int,
        template_height: int,
        top_k: int,
    ) -> list[TemplateMatchCandidate]:
        candidates: list[TemplateMatchCandidate] = []
        working = response.copy()
        suppression_radius_x = max(8, template_width // 3)
        suppression_radius_y = max(8, template_height // 3)
        for _ in range(top_k):
            _, max_value, _, max_location = cv2.minMaxLoc(working)
            if not np.isfinite(max_value):
                break
            x, y = max_location
            scale = (template_width / max(map_scale, 1e-6)) / max(frame_width, 1)
            width = max(1, int(round(frame_width * scale)))
            height = max(1, int(round(frame_height * scale)))
            x0 = map_offset_x + int(round(x / max(map_scale, 1e-6)))
            y0 = map_offset_y + int(round(y / max(map_scale, 1e-6)))
            candidates.append(
                TemplateMatchCandidate(
                    x0=x0,
                    y0=y0,
                    width=width,
                    height=height,
                    scale=scale,
                    score=float(max_value),
                )
            )
            sx0 = max(0, x - suppression_radius_x)
            sy0 = max(0, y - suppression_radius_y)
            sx1 = min(working.shape[1], x + suppression_radius_x)
            sy1 = min(working.shape[0], y + suppression_radius_y)
            working[sy0:sy1, sx0:sx1] = -1.0
        return candidates

    def _deduplicate_template_candidates(
        self,
        candidates: list[TemplateMatchCandidate],
    ) -> list[TemplateMatchCandidate]:
        deduplicated: list[TemplateMatchCandidate] = []
        for candidate in candidates:
            candidate_center = (
                candidate.x0 + candidate.width * 0.5,
                candidate.y0 + candidate.height * 0.5,
            )
            is_duplicate = False
            for existing in deduplicated:
                existing_center = (
                    existing.x0 + existing.width * 0.5,
                    existing.y0 + existing.height * 0.5,
                )
                distance = float(np.hypot(candidate_center[0] - existing_center[0], candidate_center[1] - existing_center[1]))
                size_reference = max(24.0, min(candidate.width, candidate.height, existing.width, existing.height) * 0.4)
                if distance <= size_reference:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduplicated.append(candidate)
            if len(deduplicated) >= self.template_match_top_k:
                break
        return deduplicated

    @staticmethod
    def _template_candidate_homography(candidate: TemplateMatchCandidate) -> np.ndarray:
        return np.array(
            [
                [candidate.scale, 0.0, candidate.x0],
                [0.0, candidate.scale, candidate.y0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )

    @staticmethod
    def _project_corners(frame_gray: np.ndarray, homography: np.ndarray) -> np.ndarray:
        height, width = frame_gray.shape[:2]
        frame_corners = np.float32(
            [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]]
        ).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(frame_corners, homography).reshape(-1, 2)
        return transformed

    def _build_result(
        self,
        frame_gray: np.ndarray,
        homography: np.ndarray,
        match_count: int,
        inlier_count: int,
        feature_score: float,
        verification_score: float | None,
        state: str,
        method: str,
    ) -> LocalizationResult:
        corners = self._project_corners(frame_gray, homography)
        if not self._is_geometry_plausible(frame_gray, corners):
            raise ValueError("不合理的投影几何")

        center_x = float(np.mean(corners[:, 0]))
        center_y = float(np.mean(corners[:, 1]))
        theta = self._estimate_rotation_degrees(corners)

        x_values = corners[:, 0]
        y_values = corners[:, 1]
        bbox = (
            int(np.floor(x_values.min())),
            int(np.floor(y_values.min())),
            int(np.ceil(x_values.max())),
            int(np.ceil(y_values.max())),
        )

        return LocalizationResult(
            x=center_x,
            y=center_y,
            theta=theta,
            score=self._blend_scores(feature_score, verification_score),
            state=state,
            method=method,
            matches=match_count,
            inliers=inlier_count,
            bbox=bbox,
            corners=[(float(x), float(y)) for x, y in corners],
        )

    def _try_build_result(
        self,
        frame_gray: np.ndarray,
        homography: np.ndarray,
        match_count: int,
        inlier_count: int,
        feature_score: float,
        verification_score: float | None,
        state: str,
        method: str,
    ) -> Optional[LocalizationResult]:
        try:
            return self._build_result(
                frame_gray=frame_gray,
                homography=homography,
                match_count=match_count,
                inlier_count=inlier_count,
                feature_score=feature_score,
                verification_score=verification_score,
                state=state,
                method=method,
            )
        except ValueError:
            return None

    def _build_global_tiles(self) -> list[GlobalTile]:
        if self.global_tile_size <= 0 or self.global_tile_top_k <= 0:
            return []

        height, width = self.map_gray.shape[:2]
        if width <= self.global_tile_size and height <= self.global_tile_size:
            return []

        x_positions = self._sliding_positions(width, self.global_tile_size, self.global_tile_stride)
        y_positions = self._sliding_positions(height, self.global_tile_size, self.global_tile_stride)

        tiles: list[GlobalTile] = []
        for y0 in y_positions:
            for x0 in x_positions:
                x1 = min(width, x0 + self.global_tile_size)
                y1 = min(height, y0 + self.global_tile_size)
                roi = self.map_gray[y0:y1, x0:x1]
                keypoints, descriptors = self.matcher.detect_and_compute(roi)
                if len(keypoints) < 4 or descriptors is None:
                    continue
                tiles.append(
                    GlobalTile(
                        x0=x0,
                        y0=y0,
                        x1=x1,
                        y1=y1,
                        keypoints=self.matcher.shift_keypoints(keypoints, x0, y0),
                        descriptors=descriptors,
                    )
                )
        return tiles

    @staticmethod
    def _sliding_positions(length: int, window: int, stride: int) -> list[int]:
        if window <= 0 or length <= window:
            return [0]

        stride = max(1, stride)
        positions = list(range(0, max(1, length - window + 1), stride))
        last_start = max(0, length - window)
        if positions[-1] != last_start:
            positions.append(last_start)
        return positions

    @staticmethod
    def _adjust_homography_for_scale(homography: np.ndarray, scale: float) -> np.ndarray:
        scale_matrix = np.array(
            [[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        return homography @ scale_matrix

    @staticmethod
    def _resize_frame(frame_gray: np.ndarray, scale: float) -> np.ndarray | None:
        if scale <= 0:
            return None
        if abs(scale - 1.0) < 1e-6:
            return frame_gray

        height, width = frame_gray.shape[:2]
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        if scaled_width < 48 or scaled_height < 48:
            return None

        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
        return cv2.resize(frame_gray, (scaled_width, scaled_height), interpolation=interpolation)

    @staticmethod
    def _resize_mask(mask: np.ndarray | None, scale: float) -> np.ndarray | None:
        if mask is None:
            return None
        if abs(scale - 1.0) < 1e-6:
            return mask

        height, width = mask.shape[:2]
        scaled_width = max(1, int(round(width * scale)))
        scaled_height = max(1, int(round(height * scale)))
        return cv2.resize(mask, (scaled_width, scaled_height), interpolation=cv2.INTER_NEAREST)

    @staticmethod
    def _normalize_blur_size(value: int) -> int:
        size = max(3, int(value))
        if size % 2 == 0:
            size += 1
        return size

    @staticmethod
    def _normalize_scales(frame_scales: list[float]) -> list[float]:
        normalized = []
        for raw_scale in frame_scales:
            try:
                scale = float(raw_scale)
            except (TypeError, ValueError):
                continue
            if scale <= 0:
                continue
            if not any(abs(scale - existing) < 1e-6 for existing in normalized):
                normalized.append(scale)
        return normalized or [1.0]

    @staticmethod
    def _update_best_result(
        current: Optional[LocalizationResult],
        candidate: Optional[LocalizationResult],
    ) -> Optional[LocalizationResult]:
        if candidate is None:
            return current
        if current is None or candidate.score > current.score:
            return candidate
        return current

    def _score_matches(self, match_count: int, inlier_count: int) -> float:
        inlier_ratio = inlier_count / max(match_count, 1)
        support_ratio = min(1.0, inlier_count / max(self.min_match_count * 2, 1))
        return float(0.55 * inlier_ratio + 0.45 * support_ratio)

    def _blend_scores(self, feature_score: float, verification_score: float | None) -> float:
        if verification_score is None:
            return float(feature_score)
        return float(
            (1.0 - self.candidate_verification_weight) * feature_score
            + self.candidate_verification_weight * verification_score
        )

    def _verify_candidate(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray | None,
        homography: np.ndarray,
    ) -> float | None:
        comparison_mask = self._comparison_mask(frame_gray, content_mask, homography)
        if comparison_mask is None or np.count_nonzero(comparison_mask) < 256:
            return None

        try:
            inverse_homography = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return None

        aligned_map = cv2.warpPerspective(
            self.map_gray,
            inverse_homography,
            (frame_gray.shape[1], frame_gray.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        intensity_score = self._masked_intensity_similarity(frame_gray, aligned_map, comparison_mask)
        edge_score = self._masked_edge_similarity(frame_gray, aligned_map, comparison_mask)
        return float(np.clip(0.6 * intensity_score + 0.4 * edge_score, 0.0, 1.0))

    def _comparison_mask(
        self,
        frame_gray: np.ndarray,
        content_mask: np.ndarray | None,
        homography: np.ndarray,
    ) -> np.ndarray | None:
        if content_mask is None:
            base_mask = np.full(frame_gray.shape[:2], 255, dtype=np.uint8)
        else:
            base_mask = content_mask.copy()

        try:
            inverse_homography = np.linalg.inv(homography)
        except np.linalg.LinAlgError:
            return None

        map_coverage = cv2.warpPerspective(
            np.full(self.map_gray.shape[:2], 255, dtype=np.uint8),
            inverse_homography,
            (frame_gray.shape[1], frame_gray.shape[0]),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        base_mask[map_coverage == 0] = 0
        return base_mask

    @staticmethod
    def _masked_intensity_similarity(
        frame_gray: np.ndarray,
        aligned_map: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        frame_blurred = cv2.GaussianBlur(frame_gray, (5, 5), 0)
        map_blurred = cv2.GaussianBlur(aligned_map, (5, 5), 0)

        valid = mask > 0
        if int(valid.sum()) < 256:
            return 0.0

        frame_values = frame_blurred[valid].astype(np.float32)
        map_values = map_blurred[valid].astype(np.float32)
        abs_diff = np.abs(frame_values - map_values)
        if abs_diff.size >= 64:
            cutoff = float(np.quantile(abs_diff, 0.85))
            abs_diff = abs_diff[abs_diff <= cutoff]
        trimmed_similarity = 1.0 - float(abs_diff.mean() / 255.0)

        frame_centered = frame_values - float(frame_values.mean())
        map_centered = map_values - float(map_values.mean())
        denominator = float(np.linalg.norm(frame_centered) * np.linalg.norm(map_centered))
        correlation = 0.0 if denominator <= 1e-6 else float(np.dot(frame_centered, map_centered) / denominator)
        correlation = (correlation + 1.0) * 0.5
        return float(np.clip(0.55 * trimmed_similarity + 0.45 * correlation, 0.0, 1.0))

    @staticmethod
    def _masked_edge_similarity(
        frame_gray: np.ndarray,
        aligned_map: np.ndarray,
        mask: np.ndarray,
    ) -> float:
        frame_edges = cv2.Canny(frame_gray, 60, 160)
        map_edges = cv2.Canny(aligned_map, 60, 160)
        frame_edges[mask == 0] = 0
        map_edges[mask == 0] = 0

        edge_pixels = int(np.count_nonzero(frame_edges) + np.count_nonzero(map_edges))
        if edge_pixels < 40:
            return 0.5

        frame_edges = cv2.dilate(frame_edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        map_edges = cv2.dilate(map_edges, np.ones((3, 3), dtype=np.uint8), iterations=1)
        overlap = int(np.count_nonzero((frame_edges > 0) & (map_edges > 0)))
        total = max(1, int(np.count_nonzero(frame_edges) + np.count_nonzero(map_edges)))
        return float(np.clip((2.0 * overlap) / total, 0.0, 1.0))

    def _is_geometry_plausible(self, frame_gray: np.ndarray, corners: np.ndarray) -> bool:
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return False

        contour = corners.astype(np.float32)
        if not cv2.isContourConvex(contour):
            return False

        x_values = corners[:, 0]
        y_values = corners[:, 1]
        bbox_width = float(x_values.max() - x_values.min())
        bbox_height = float(y_values.max() - y_values.min())
        polygon_area = float(abs(cv2.contourArea(contour)))

        min_extent = max(8.0, min(frame_gray.shape[:2]) * 0.05)
        min_area = max(64.0, min_extent * min_extent)
        if bbox_width < min_extent or bbox_height < min_extent:
            return False
        if polygon_area < min_area:
            return False

        side_vectors = np.roll(corners, -1, axis=0) - corners
        side_lengths = np.linalg.norm(side_vectors, axis=1)
        if float(side_lengths.min()) < min_extent:
            return False

        top, right, bottom, left = [float(value) for value in side_lengths]
        width = (top + bottom) * 0.5
        height = (right + left) * 0.5
        if height <= 1e-6 or width <= 1e-6:
            return False

        frame_aspect = frame_gray.shape[1] / max(frame_gray.shape[0], 1)
        candidate_aspect = width / height
        aspect_ratio = max(candidate_aspect / frame_aspect, frame_aspect / candidate_aspect)
        if aspect_ratio > 3.0:
            return False

        if max(top, bottom) / max(min(top, bottom), 1e-6) > 4.0:
            return False
        if max(left, right) / max(min(left, right), 1e-6) > 4.0:
            return False

        fill_ratio = polygon_area / max(bbox_width * bbox_height, 1.0)
        if fill_ratio < 0.2:
            return False

        if not self._is_rotation_plausible(corners):
            return False
        return True

    @staticmethod
    def _normalize_angle_degrees(angle: float) -> float:
        normalized = (angle + 180.0) % 360.0 - 180.0
        return float(normalized)

    def _estimate_rotation_degrees(self, corners: np.ndarray) -> float:
        top_edge = corners[1] - corners[0]
        bottom_edge = corners[2] - corners[3]
        direction = top_edge + bottom_edge
        if float(np.linalg.norm(direction)) <= 1e-6:
            direction = top_edge

        angle = math.degrees(math.atan2(direction[1], direction[0]))
        return self._normalize_angle_degrees(angle)

    def _is_rotation_plausible(self, corners: np.ndarray) -> bool:
        if self.max_rotation_degrees <= 0:
            return True

        top_edge = corners[1] - corners[0]
        bottom_edge = corners[2] - corners[3]
        top_angle = self._normalize_angle_degrees(math.degrees(math.atan2(top_edge[1], top_edge[0])))
        bottom_angle = self._normalize_angle_degrees(math.degrees(math.atan2(bottom_edge[1], bottom_edge[0])))
        estimated_angle = self._estimate_rotation_degrees(corners)

        if abs(estimated_angle) > self.max_rotation_degrees:
            return False
        if abs(self._normalize_angle_degrees(top_angle - bottom_angle)) > self.max_rotation_degrees * 1.5:
            return False
        return True
