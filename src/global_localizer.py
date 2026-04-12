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
    ) -> None:
        self.map_gray = map_gray
        self.matcher = matcher
        self.min_match_count = min_match_count
        self.ransac_threshold = ransac_threshold
        self.frame_scales = self._normalize_scales(frame_scales or [1.0])
        self.global_tile_size = global_tile_size
        self.global_tile_stride = global_tile_stride
        self.global_tile_top_k = global_tile_top_k
        self.map_keypoints, self.map_descriptors = matcher.detect_and_compute(map_gray)
        self.global_tiles = self._build_global_tiles()

    def localize(
        self,
        frame_gray: np.ndarray,
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
        for scale in self.frame_scales:
            scaled_frame = self._resize_frame(frame_gray, scale)
            if scaled_frame is None:
                continue

            frame_keypoints, frame_descriptors = self.matcher.detect_and_compute(scaled_frame)
            if len(frame_keypoints) < 4 or frame_descriptors is None:
                continue

            if search_region is None:
                tile_result = self._search_global_tiles(
                    frame_gray=frame_gray,
                    frame_keypoints=frame_keypoints,
                    frame_descriptors=frame_descriptors,
                    scale=scale,
                    state=state,
                )
                best_result = self._update_best_result(best_result, tile_result)

            direct_result = self._localize_from_descriptors(
                frame_gray=frame_gray,
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
        return self._try_build_result(
            frame_gray=frame_gray,
            homography=adjusted_homography,
            match_count=len(matches),
            inlier_count=inliers,
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
        state: str,
        method: str,
    ) -> LocalizationResult:
        corners = self._project_corners(frame_gray, homography)
        if not self._is_geometry_plausible(frame_gray, corners):
            raise ValueError("不合理的投影几何")

        center_x = float(np.mean(corners[:, 0]))
        center_y = float(np.mean(corners[:, 1]))
        theta = math.degrees(
            math.atan2(corners[1, 1] - corners[0, 1], corners[1, 0] - corners[0, 0])
        )

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
            score=self._score_matches(match_count, inlier_count),
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
        state: str,
        method: str,
    ) -> Optional[LocalizationResult]:
        try:
            return self._build_result(
                frame_gray=frame_gray,
                homography=homography,
                match_count=match_count,
                inlier_count=inlier_count,
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

    @staticmethod
    def _is_geometry_plausible(frame_gray: np.ndarray, corners: np.ndarray) -> bool:
        if corners.shape != (4, 2) or not np.isfinite(corners).all():
            return False

        x_values = corners[:, 0]
        y_values = corners[:, 1]
        bbox_width = float(x_values.max() - x_values.min())
        bbox_height = float(y_values.max() - y_values.min())
        polygon_area = float(abs(cv2.contourArea(corners.astype(np.float32))))

        min_extent = max(8.0, min(frame_gray.shape[:2]) * 0.05)
        min_area = max(64.0, min_extent * min_extent)
        if bbox_width < min_extent or bbox_height < min_extent:
            return False
        if polygon_area < min_area:
            return False
        return True
