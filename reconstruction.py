import numpy as np
import cv2
from camera import Camera, estimate_pose_from_essential, triangulate_points
from features import FeatureDetector, FeatureMatcher


class RoomReconstructor:
    """Реконструкция комнаты из фотографий."""

    def __init__(self, K):
        self.K = K
        self.detector = FeatureDetector(max_features=3000)
        self.matcher = FeatureMatcher(ratio_threshold=0.75)
        self.cameras = []
        self.points_3d = []

    def reconstruct(self, images):
        print(f"Обработка {len(images)} изображений...")

        # 1. Детекция ключевых точек на всех изображениях
        all_keypoints = []
        all_descriptors = []

        for i, img in enumerate(images):
            kp, desc = self.detector.detect(img)
            all_keypoints.append(kp)
            all_descriptors.append(desc)
            print(f"  Изображение {i+1}: {len(kp)} ключевых точек")

        # 2. Сопоставление между первой парой
        print("\nСопоставление первой пары изображений...")
        matches = self.matcher.match(all_descriptors[0], all_descriptors[1])
        print(f"  Найдено {len(matches)} соответствий")

        # Фильтрация геометрии
        matches, F = self.matcher.filter_by_geometry(
            all_keypoints[0], all_keypoints[1], matches, self.K
        )
        print(f" После геометрической фильтрации: {len(matches)}")

        if len(matches) < 20:
            raise ValueError("Слишком мало соответствий для реконструкции")

        # 3. Оценка позы
        pts1 = np.float32([all_keypoints[0][m.queryIdx].pt for m in matches])
        pts2 = np.float32([all_keypoints[1][m.trainIdx].pt for m in matches])

        # Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, self.K, method=cv2.RANSAC, prob=0.999, threshold=1.0)

        # Восстановление позы
        R, t = estimate_pose_from_essential(E, self.K, pts1, pts2)

        # 4. Создание камер
        cam1 = Camera(self.K)
        cam1.set_pose(np.eye(3), np.zeros(3))

        cam2 = Camera(self.K)
        cam2.set_pose(R, t)

        self.cameras = [cam1, cam2]

        # 5. Триангуляция точек
        inlier_pts1 = pts1[mask.ravel() > 0]
        inlier_pts2 = pts2[mask.ravel() > 0]

        points_3d = triangulate_points(cam1, cam2, inlier_pts1, inlier_pts2)

        # Фильтрация выбросов по глубине
        z_values = points_3d[:, 2]
        valid_mask = (z_values > 0) & (z_values < 50)
        points_3d = points_3d[valid_mask]

        self.points_3d = points_3d

        print(f"\nРеконструировано {len(points_3d)} 3D точек")

        return points_3d, self.cameras
