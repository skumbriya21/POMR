import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional
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


@dataclass
class Door:
    """Дверной проем."""
    x: float  # Позиция вдоль стены от начала
    width: float  # Ширина двери
    height: float  # Высота двери
    wall: str  # На какой стене: 'left', 'right', 'top', 'bottom'
    has_glass: bool = False  # Стеклянная ли дверь
    is_open: bool = False  # Открыта ли дверь на фото
    confidence: float = 1.0  # Уверенность детекции


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
        """Автоматическое размещение окон."""
        windows = []

        area = room_dims.area

        if area < 12:
            num_windows = 1
        elif area < 24:
            num_windows = random.choice([1, 2])
        elif area < 40:
            num_windows = 2
        else:
            num_windows = random.choice([2, 3])

        window_width = random.choice([1.0, 1.2, 1.5])
        window_height = random.choice([1.2, 1.4, 1.5])

        walls = ['right', 'top', 'left', 'bottom']

        for i in range(num_windows):
            if num_windows <= 2:
                wall = walls[i % 2]
            else:
                wall = walls[i % 4]

            if wall in ['left', 'right']:
                max_pos = room_dims.length
                margin = room_dims.length * 0.15
                x = random.uniform(margin, max_pos - margin - window_width)
            else:
                max_pos = room_dims.width
                margin = room_dims.width * 0.15
                x = random.uniform(margin, max_pos - margin - window_width)

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

            return RoomDimensions(width, length, height, width * length), None
        except ValueError:
            print("Ошибка ввода!")
            return None, None

    elif choice == "3":
        try:
            print("\nУкажите один известный размер:")
            kw = input("Известная ширина (м) [Enter - неизвестно]: ").strip()
            kl = input("Известная длина (м) [Enter - неизвестно]: ").strip()

            known_width = float(kw) if kw else None
            known_length = float(kl) if kl else None

            return None, (known_width, known_length)
        except ValueError:
            print("Ошибка ввода!")
            return None, None

    return None, None


def auto_place_door(room_dims: RoomDimensions, windows: List[Window],
                    detected_door_info: Optional[Dict] = None) -> Door:
    """
    Автоматическое размещение двери на основе детекции с фото.

    Args:
        room_dims: Размеры комнаты
        windows: Список окон
        detected_door_info: Информация о двери с фото (из DoorDetector)

    Returns:
        Объект Door
    """
    w, l = room_dims.width, room_dims.length

    # Если есть детекция с фото - используем её
    if detected_door_info:
        wall = detected_door_info.get('wall', 'bottom')
        door_x = detected_door_info.get('x', w * 0.5)
        door_width = detected_door_info.get('width', 0.9)
        door_height = detected_door_info.get('height', 2.0)
        has_glass = detected_door_info.get('has_glass', False)
        is_open = detected_door_info.get('is_open', False)
        confidence = detected_door_info.get('confidence', 1.0)

        return Door(
            x=round(door_x, 2),
            width=door_width,
            height=door_height,
            wall=wall,
            has_glass=has_glass,
            is_open=is_open,
            confidence=confidence
        )

    # Иначе используем эвристическое размещение
    walls_with_windows = set(win.wall for win in windows)

    # Выбираем стену для двери
    if w >= l:
        long_walls = ['top', 'bottom']
        short_walls = ['left', 'right']
        max_pos = w
    else:
        long_walls = ['left', 'right']
        short_walls = ['top', 'bottom']
        max_pos = l

    door_wall = None
    for wall in long_walls:
        if wall not in walls_with_windows:
            door_wall = wall
            break

    if door_wall is None:
        door_wall = long_walls[0]

    if door_wall in walls_with_windows:
        for wall in short_walls:
            if wall not in walls_with_windows:
                door_wall = wall
                max_pos = w if wall in ['top', 'bottom'] else l
                break

    door_width = 0.9
    door_height = 2.0

    wall_windows = [win for win in windows if win.wall == door_wall]

    # Поиск свободного места
    left_position = 0.15
    right_position = max_pos - door_width - 0.15

    if not wall_windows:
        door_x = left_position
    else:
        sorted_windows = sorted(wall_windows, key=lambda win: win.x)
        first_window_start = sorted_windows[0].x

        if first_window_start >= door_width + 0.3:
            door_x = left_position
        else:
            last_window = sorted_windows[-1]
            last_window_end = last_window.x + last_window.width

            if max_pos - last_window_end >= door_width + 0.3:
                door_x = right_position
            else:
                found_spot = False
                best_x = left_position

                for i in range(len(sorted_windows) - 1):
                    gap_start = sorted_windows[i].x + sorted_windows[i].width
                    gap_end = sorted_windows[i + 1].x
                    gap_size = gap_end - gap_start

                    if gap_size >= door_width + 0.4:
                        best_x = gap_start + 0.2
                        found_spot = True
                        break

                if not found_spot:
                    left_space = first_window_start
                    right_space = max_pos - last_window_end

                    if left_space >= right_space:
                        door_x = max(0.1, first_window_start - door_width - 0.1)
                    else:
                        door_x = min(max_pos - door_width - 0.1, last_window_end + 0.1)
                else:
                    door_x = best_x

    door_x = max(0.1, min(door_x, max_pos - door_width - 0.1))

    print(f"  Дверь размещена эвристически: стена {door_wall}, позиция {door_x:.2f}м")

    return Door(
        x=round(door_x, 2),
        width=door_width,
        height=door_height,
        wall=door_wall,
        has_glass=False,
        is_open=False,
        confidence=0.5
    )