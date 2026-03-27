import cv2
import numpy as np


class FeatureDetector:
    """Детектор ключевых точек SIFT."""

    def __init__(self, max_features=5000):
        self.detector = cv2.SIFT_create(nfeatures=max_features)

    def detect(self, image):
        """Детекция ключевых точек."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return keypoints, descriptors


class FeatureMatcher:
    """Сопоставление ключевых точек."""

    def __init__(self, ratio_threshold=0.7):
        self.matcher = cv2.BFMatcher(cv2.NORM_L2)
        self.ratio_threshold = ratio_threshold

    def match(self, desc1, desc2):
        """Сопоставление дескрипторов."""
        if desc1 is None or desc2 is None:
            return []

        # k-NN поиск
        matches = self.matcher.knnMatch(desc1, desc2, k=2)

        # Ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.ratio_threshold * n.distance:
                    good_matches.append(m)

        return good_matches

    def filter_by_geometry(self, kp1, kp2, matches, K, threshold=3.0):
        """Фильтрация с помощью Fundamental matrix."""
        if len(matches) < 8:
            return matches, None

        pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

        # Находим F с помощью RANSAC
        F, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_RANSAC, threshold)

        if mask is None:
            return matches, None

        mask = mask.ravel().astype(bool)
        inliers = [m for m, valid in zip(matches, mask) if valid]

        return inliers, F
