import numpy as np
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class RoomDimensions:
    """Размеры комнаты."""
    width: float      # Ширина (метры)
    length: float     # Длина (метры)
    height: float     # Высота (метры)
    area: float       # Площадь (м²)


@dataclass
class Window:
    """Окно в комнате."""
    x: float          # Позиция по X (относительно комнаты)
    y: float          # Позиция по Y (высота от пола)
    width: float      # Ширина окна
    height: float     # Высота окна
    wall: str         # Стена: 'left', 'right', 'top', 'bottom'


class RoomDetector:
    """Определение геометрии комнаты."""

    def __init__(self):
        self.wall_height = 2.7  # Стандартная высота потолка

    def detect_room(self, points_3d):
        if len(points_3d) == 0:
            raise ValueError("Пустое облако точек")

        # Находим bounding box
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)

        # Размеры по осям
        x_size = max_coords[0] - min_coords[0]
        y_size = max_coords[1] - min_coords[1]  # Высота
        z_size = max_coords[2] - min_coords[2]

        # Определяем ширину и длину (X и Z - горизонтальные оси)
        width = max(x_size, z_size)
        length = min(x_size, z_size)
        height = y_size if y_size > 1.5 else self.wall_height

        # Корректировка масштаба (если комната слишком маленькая или большая)
        if width < 2 or width > 20:
            scale_factor = 5.0 / width if width > 0 else 1.0
            width *= scale_factor
            length *= scale_factor
            height *= scale_factor

        area = width * length

        return RoomDimensions(
            width=round(width, 2),
            length=round(length, 2),
            height=round(height, 2),
            area=round(area, 2)
        )

    def detect_windows(self, points_3d, images=None):
        windows = []

        # Упрощенная эвристика: предполагаем 1-2 окна на стенах
        # В реальности здесь должен быть анализ плотности точек

        room_dims = self.detect_room(points_3d)

        # Добавляем "фиктивное" окно для демонстрации
        if len(points_3d) > 100:
            windows.append(Window(
                x=room_dims.width * 0.3,
                y=1.0,  # Высота от пола
                width=1.2,
                height=1.5,
                wall='right'
            ))

            # Второе окно с вероятностью 50%
            if len(points_3d) > 500:
                windows.append(Window(
                    x=room_dims.length * 0.4,
                    y=1.0,
                    width=1.0,
                    height=1.2,
                    wall='top'
                ))

        return windows

    def get_room_bounds(self, points_3d):
        """Получение границ комнаты."""
        min_coords = np.min(points_3d, axis=0)
        max_coords = np.max(points_3d, axis=0)
        return min_coords, max_coords


def ask_room_dimensions():
    print("\n~УКАЖИТЕ РАЗМЕРЫ КОМНАТЫ~")
    print("1. Автоопределение из фотографий")
    print("2. Ввести размеры вручную")
    print()

    choice = input("Выберите вариант (1 или 2): ").strip()

    if choice == "2":
        try:
            width = float(input("Ширина комнаты (м): "))
            length = float(input("Длина комнаты (м): "))
            height = float(input("Высота потолка (м) [2.7]: ") or "2.7")

            return RoomDimensions(
                width=width,
                length=length,
                height=height,
                area=width * length
            )
        except ValueError:
            print("Ошибка ввода! Используем автоопределение.")
            return None

    return None  # Автоопределение

