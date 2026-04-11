import numpy as np
import cv2


class Camera:
    """Камера с внутренними и внешними параметрами."""

    def __init__(self, K, dist_coeffs=None):
        self.K = K
        self.dist_coeffs = dist_coeffs if dist_coeffs is not None else np.zeros(5)
        self.R = np.eye(3)
        self.t = np.zeros((3, 1))

    def set_pose(self, R, t):
        """Установка позы камеры."""
        self.R = R
        self.t = t.reshape((3, 1))

    def project(self, points_3d):
        """Проецирование 3D точек на изображение."""
        # Преобразование в систему координат камеры
        points_cam = (self.R @ points_3d.T + self.t).T

        # Только точки перед камерой
        valid = points_cam[:, 2] > 0

        # Проекция
        points_2d_hom = (self.K @ points_cam.T).T
        points_2d = points_2d_hom[:, :2] / points_2d_hom[:, 2:3]

        return points_2d, valid

    def get_projection_matrix(self):
        """Матрица проекции P = K[R|t]."""
        Rt = np.hstack([self.R, self.t])
        return self.K @ Rt


def estimate_pose_from_essential(E, K, pts1, pts2):
    """Оценка позы из Essential matrix."""
    # Декомпозиция E
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Проверяем 4 варианта
    best_R, best_t = R1, t
    max_positive = 0

    for R in [R1, R2]:
        for t_vec in [t, -t]:
            # Триангуляция тестовых точек
            P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
            P2 = K @ np.hstack([R, t_vec])

            pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
            pts3d = pts4d[:3] / pts4d[3]

            # Считаем точки перед обеими камерами
            z1 = pts3d[2, :]
            z2 = (R[2, :] @ pts3d + t_vec[2, 0])
            positive = np.sum((z1 > 0) & (z2 > 0))

            if positive > max_positive:
                max_positive = positive
                best_R = R
                best_t = t_vec

    return best_R, best_t


def triangulate_points(cam1, cam2, pts1, pts2):
    """Триангуляция 3D точек."""
    P1 = cam1.get_projection_matrix()
    P2 = cam2.get_projection_matrix()

    pts4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    pts3d = (pts4d[:3] / pts4d[3]).T

    return pts3d
