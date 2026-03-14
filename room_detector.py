import numpy as np
from dataclasses import dataclass
from typing import List
import random


@dataclass
class RoomDimensions:
    """Размеры комнаты."""
    width: float
    length: float
    height: float
    area: float


@dataclass
class Window:
    """Окно в комнате."""
    x: float
    y: float
    width: float
    height: float
    wall: str


class RoomDetector:
    """Определение геометрии комнаты."""

    def __init__(self, known_width=None, known_length=None):
        self.known_width = known_width
        self.known_length = known_length
        self.wall_height = 2.7

    def detect_room(self, points_3d):
        """Определение размеров комнаты."""
        if len(points_3d) == 0:
            raise ValueError("Пустое облако точек")

        points_clean = self._remove_outliers(points_3d)

        min_coords = np.min(points_clean, axis=0)
        max_coords = np.max(points_clean, axis=0)

        x_size = max_coords[0] - min_coords[0]
        y_size = max_coords[1] - min_coords[1]
        z_size = max_coords[2] - min_coords[2]

        width = max(x_size, z_size)
        length = min(x_size, z_size)
        height = y_size

        # Калибровка
        if self.known_width and width > 0:
            scale = self.known_width / width
            width *= scale
            length *= scale
            height *= scale
        elif self.known_length and length > 0:
            scale = self.known_length / length
            width *= scale
            length *= scale
            height *= scale
        else:
            if 1.5 < height < 5:
                scale = self.wall_height / height
                width *= scale
                length *= scale
                height = self.wall_height

        # Ограничения
        width = max(min(width, 15), 2)
        length = max(min(length, 15), 2)
        height = max(min(height, 4), 2.4)

        return RoomDimensions(
            width=round(width, 2),
            length=round(length, 2),
            height=round(height, 2),
            area=round(width * length, 2)
        )

    def _remove_outliers(self, points, threshold=2.0):
        """Удаление выбросов."""
        if len(points) < 10:
            return points

        mask = np.ones(len(points), dtype=bool)
        for i in range(3):
            median = np.median(points[:, i])
            std = np.std(points[:, i])
            if std > 0:
                mask &= np.abs(points[:, i] - median) < threshold * std
        return points[mask]

    def auto_place_windows(self, room_dims: RoomDimensions) -> List[Window]:
        """
        Автоматическое размещение окон на основе размеров комнаты.

        Логика:
        - Комнаты < 4м: 1 окно
        - Комнаты 4-6м: 1-2 окна
        - Комнаты 6-10м: 2 окна
        - Комнаты > 10м: 2-3 окна
        """
        windows = []

        # Определяем количество окон по площади
        area = room_dims.area

        if area < 12:
            num_windows = 1
        elif area < 24:
            num_windows = random.choice([1, 2])
        elif area < 40:
            num_windows = 2
        else:
            num_windows = random.choice([2, 3])

        # Размеры окон (стандартные)
        window_width = random.choice([1.0, 1.2, 1.5])
        window_height = random.choice([1.2, 1.4, 1.5])

        # Размещаем окна
        walls = ['right', 'top', 'left', 'bottom']

        for i in range(num_windows):
            # Выбираем стену
            if num_windows <= 2:
                wall = walls[i % 2]  # right и top
            else:
                wall = walls[i % 4]

            # Позиция вдоль стены (случайная, но не у края)
            if wall in ['left', 'right']:
                max_pos = room_dims.length
                margin = room_dims.length * 0.15
                x = random.uniform(margin, max_pos - margin - window_width)
            else:
                max_pos = room_dims.width
                margin = room_dims.width * 0.15
                x = random.uniform(margin, max_pos - margin - window_width)

            # Высота от пола (стандартная)
            y = random.uniform(0.8, 1.2)

            windows.append(Window(
                x=round(x, 2),
                y=round(y, 2),
                width=window_width,
                height=window_height,
                wall=wall
            ))

        return windows

    def detect_windows(self, points_3d, images=None):
        """Устаревший метод - используйте auto_place_windows."""
        return []


def ask_room_dimensions():
    """Запрос размеров у пользователя."""
    print("УКАЖИТЕ РАЗМЕРЫ КОМНАТЫ")
    print("1. Автоопределение из фотографий")
    print("2. Ввести размеры вручную (точно)")
    print("3. Указать один известный размер для калибровки")
    print()

    choice = input("Выберите вариант (1, 2 или 3): ").strip()

    if choice == "2":
        try:
            width = float(input("Ширина комнаты (м): "))
            length = float(input("Длина комнаты (м): "))
            height = float(input("Высота потолка (м) [2.7]: ") or "2.7")

            return RoomDimensions(width, length, height, width*length), None
        except ValueError:
            print("Ошибка ввода!")
            return None, None

    elif choice == "3":
        try:
            print("\\nУкажите один известный размер:")
            kw = input("Известная ширина (м) [Enter - неизвестно]: ").strip()
            kl = input("Известная длина (м) [Enter - неизвестно]: ").strip()

            known_width = float(kw) if kw else None
            known_length = float(kl) if kl else None

            return None, (known_width, known_length)
        except ValueError:
            print("Ошибка ввода!")
            return None, None

    return None, None
