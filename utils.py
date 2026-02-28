import numpy as np
import cv2


def load_images(image_paths):
    """Загрузка изображений."""
    images = []
    for path in image_paths:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
    return images


def estimate_camera_matrix(image_shape, fov_degrees=60):
    """Оценка матрицы камеры из разрешения и угла обзора."""
    h, w = image_shape[:2]
    focal_length = max(w, h) / (2 * np.tan(np.radians(fov_degrees) / 2))

    K = np.array([
        [focal_length, 0, w / 2],
        [0, focal_length, h / 2],
        [0, 0, 1]
    ])
    return K
