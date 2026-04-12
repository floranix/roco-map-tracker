from __future__ import annotations

import cv2

from src.utils import AppConfig


class FeatureMatcher:
    def __init__(self, config: AppConfig) -> None:
        self.config = config
        self.detector = self._create_detector(config.feature_type.lower())
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def detect_and_compute(self, image, mask=None):
        keypoints, descriptors = self.detector.detectAndCompute(image, mask)
        return keypoints or [], descriptors

    def match(self, query_descriptors, train_descriptors):
        if query_descriptors is None or train_descriptors is None:
            return []
        if len(query_descriptors) < 2 or len(train_descriptors) < 2:
            return []

        knn_matches = self.matcher.knnMatch(query_descriptors, train_descriptors, k=2)
        good_matches = []
        for pair in knn_matches:
            if len(pair) < 2:
                continue
            best, runner_up = pair
            if best.distance < self.config.ratio_test * runner_up.distance:
                good_matches.append(best)
        good_matches.sort(key=lambda match: match.distance)
        return good_matches

    @staticmethod
    def shift_keypoints(keypoints, offset_x: int, offset_y: int):
        shifted = []
        for kp in keypoints:
            shifted.append(
                cv2.KeyPoint(
                    kp.pt[0] + offset_x,
                    kp.pt[1] + offset_y,
                    kp.size,
                    kp.angle,
                    kp.response,
                    kp.octave,
                    kp.class_id,
                )
            )
        return shifted

    def _create_detector(self, feature_type: str):
        if feature_type == "orb":
            return cv2.ORB_create(nfeatures=self.config.orb_nfeatures)
        if feature_type == "akaze":
            return cv2.AKAZE_create()
        raise ValueError(f"Unsupported feature type: {self.config.feature_type}")
