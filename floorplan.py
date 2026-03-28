import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np
import cv2
from collections import Counter


class FloorplanDrawer:
    """Рисование планировки комнаты."""

    def __init__(self, pixels_per_meter=100):
        self.pixels_per_meter = pixels_per_meter

    def draw(self, room_dims, windows, doors=None, output_path=None, images=None):
        # Размеры изображения
        margin = 1.0  # метры
        total_width = room_dims.width + 2 * margin
        total_length = room_dims.length + 2 * margin

        fig_width = total_width * self.pixels_per_meter / 100
        fig_height = total_length * self.pixels_per_meter / 100

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Определяем цвет стен из фотографий
        wall_color = self._get_dominant_wall_color(images) if images else 'lightgray'

        # Рисуем комнату (стены)
        room_rect = Rectangle(
            (margin, margin),
            room_dims.width,
            room_dims.length,
            linewidth=3,
            edgecolor='black',
            facecolor=wall_color,
            alpha=0.3
        )
        ax.add_patch(room_rect)

        # Рисуем окна
        for window in windows:
            self._draw_window(ax, window, margin, room_dims)

        # Рисуем двери
        if doors:
            for door in doors:
                self._draw_door(ax, door, margin, room_dims)

        # Добавляем размеры
        self._add_dimensions(ax, room_dims, margin)

        # Настройка осей
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_length)
        ax.set_aspect('equal')
        ax.axis('off')

        # Заголовок
        title_text = f"Планировка комнаты\n{room_dims.width}м × {room_dims.length}м = {room_dims.area}м²"

        # Добавляем информацию о дверях в заголовок
        if doors:
            door_info = []
            for door in doors:
                door_info.append(f"{door.wall} стена")
            title_text += f"\nДвери: {', '.join(door_info)}"

        plt.title(title_text, fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                        facecolor='white', edgecolor='none')
            print(f"\nПланировка сохранена: {output_path}")
        else:
            plt.show()

        plt.close()

    def _get_dominant_wall_color(self, images):
        """Определение доминирующего цвета стен из фотографий."""
        all_colors = []

        for img in images:
            if img is None:
                continue

            h, w = img.shape[:2]

            # Берем края изображения (где обычно стены)
            edge_width = w // 8

            # Левый край
            left_edge = img[:, :edge_width]
            # Правый край
            right_edge = img[:, -edge_width:]
            # Верхний край (потолок)
            top_edge = img[:edge_width, :]

            # Собираем цвета с краев
            for edge in [left_edge, right_edge, top_edge]:
                # Уменьшаем для скорости
                small = cv2.resize(edge, (50, 50))
                # Конвертируем в RGB
                rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
                # Фильтруем темные и светлые пиксели
                pixels = rgb.reshape(-1, 3)
                brightness = np.mean(pixels, axis=1)
                # Берем пиксели со средней яркостью (не тени, не блики)
                mask = (brightness > 50) & (brightness < 200)
                valid_pixels = pixels[mask]

                if len(valid_pixels) > 0:
                    # Группируем похожие цвета
                    quantized = (valid_pixels // 30) * 30 + 15
                    all_colors.extend(quantized)

        if not all_colors:
            return 'lightgray'

        # Находим самый частый цвет
        color_counts = Counter(map(tuple, all_colors))
        most_common = color_counts.most_common(1)[0][0]

        # Конвертируем в hex для matplotlib
        hex_color = '#{:02x}{:02x}{:02x}'.format(most_common[0], most_common[1], most_common[2])

        print(f"  Определен доминирующий цвет стен: {hex_color}")
        return hex_color

    def _draw_window(self, ax, window, margin, room_dims):
        """Рисование окна."""
        # Определяем позицию окна на стене
        if window.wall == 'right':
            x = margin + room_dims.width - 0.15  # Небольшой отступ от стены
            y = margin + window.x
            width = 0.3
            height = window.width

        elif window.wall == 'left':
            x = margin - 0.15
            y = margin + window.x
            width = 0.3
            height = window.width

        elif window.wall == 'top':
            x = margin + window.x
            y = margin + room_dims.length - 0.15
            width = window.width
            height = 0.3

        else:  # bottom
            x = margin + window.x
            y = margin - 0.15
            width = window.width
            height = 0.3

        # Рисуем окно
        window_rect = Rectangle(
            (x, y),
            width,
            height,
            facecolor='lightblue',
            edgecolor='blue',
            linewidth=2,
            alpha=0.8
        )
        ax.add_patch(window_rect)

        # Добавляем перекладины для окна (крестик)
        if window.wall in ['left', 'right']:
            # Вертикальное окно
            center_x = x + width / 2
            center_y = y + height / 2
            # Горизонтальная линия
            ax.plot([x, x + width], [center_y, center_y],
                    color='white', linewidth=1, alpha=0.7)
            # Вертикальная линия
            ax.plot([center_x, center_x], [y, y + height],
                    color='white', linewidth=1, alpha=0.7)
        else:
            # Горизонтальное окно
            center_x = x + width / 2
            center_y = y + height / 2
            # Горизонтальная линия
            ax.plot([x, x + width], [center_y, center_y],
                    color='white', linewidth=1, alpha=0.7)
            # Вертикальная линия
            ax.plot([center_x, center_x], [y, y + height],
                    color='white', linewidth=1, alpha=0.7)

    def _draw_door(self, ax, door, margin, room_dims):
        """Рисование двери."""
        # Определяем позицию двери на стене
        door_width_draw = 0.15  # Толщина линии двери на плане

        # Определяем цвет двери (используем стандартные цвета matplotlib)
        if door.has_glass:
            facecolor = 'lightblue'
            edgecolor = 'blue'
            alpha = 0.7
        else:
            facecolor = 'saddlebrown'  # Стандартный цвет в matplotlib
            edgecolor = 'chocolate'  # Стандартный цвет в matplotlib
            alpha = 0.9

        if door.wall == 'right':
            x = margin + room_dims.width - door_width_draw
            y = margin + door.x
            width = door_width_draw
            height = door.width

            # Рисуем дверное полотно (дуга открывания)
            if door.is_open:
                # Дуга открывания
                theta = np.linspace(0, np.pi / 2, 30)
                arc_x = x + door_width_draw + door.width * np.cos(theta)
                arc_y = y + door.width * np.sin(theta)
                ax.plot(arc_x, arc_y, color='goldenrod', linewidth=1.5, alpha=0.7)

        elif door.wall == 'left':
            x = margin - door_width_draw
            y = margin + door.x
            width = door_width_draw
            height = door.width

            if door.is_open:
                theta = np.linspace(np.pi / 2, np.pi, 30)
                arc_x = x + door_width_draw + door.width * np.cos(theta)
                arc_y = y + door.width * np.sin(theta)
                ax.plot(arc_x, arc_y, color='goldenrod', linewidth=1.5, alpha=0.7)

        elif door.wall == 'top':
            x = margin + door.x
            y = margin + room_dims.length - door_width_draw
            width = door.width
            height = door_width_draw

            if door.is_open:
                theta = np.linspace(np.pi / 2, np.pi, 30)
                arc_x = x + door.width * np.cos(theta - np.pi / 2)
                arc_y = y + door_width_draw + door.width * np.sin(theta - np.pi / 2)
                ax.plot(arc_x, arc_y, color='goldenrod', linewidth=1.5, alpha=0.7)

        else:  # bottom
            x = margin + door.x
            y = margin - door_width_draw
            width = door.width
            height = door_width_draw

            if door.is_open:
                theta = np.linspace(0, np.pi / 2, 30)
                arc_x = x + door.width * np.cos(theta)
                arc_y = y + door_width_draw + door.width * np.sin(theta)
                ax.plot(arc_x, arc_y, color='goldenrod', linewidth=1.5, alpha=0.7)

        # Рисуем дверь
        door_rect = Rectangle(
            (x, y),
            width,
            height,
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=2,
            alpha=alpha
        )
        ax.add_patch(door_rect)

        # Добавляем символ ручки
        if door.wall == 'right':
            handle_x = x - 0.05
            handle_y = y + door.width / 2
        elif door.wall == 'left':
            handle_x = x + width + 0.05
            handle_y = y + door.width / 2
        elif door.wall == 'top':
            handle_x = x + door.width / 2
            handle_y = y - 0.05
        else:  # bottom
            handle_x = x + door.width / 2
            handle_y = y + height + 0.05

        # Рисуем ручку
        ax.plot(handle_x, handle_y, 'o', color='gold', markersize=3)

        # Добавляем текст для стеклянной двери
        if door.has_glass:
            if door.wall in ['left', 'right']:
                text_x = x + width / 2
                text_y = y + door.width / 2
            else:
                text_x = x + door.width / 2
                text_y = y + height / 2
            ax.text(text_x, text_y, '🪟', fontsize=8, ha='center', va='center', alpha=0.6)

    def _add_dimensions(self, ax, room_dims, margin):
        """Добавление размеров на чертеж."""
        # Ширина (сверху)
        ax.annotate('',
                    xy=(margin + room_dims.width, margin + room_dims.length + 0.3),
                    xytext=(margin, margin + room_dims.length + 0.3),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

        ax.text(margin + room_dims.width / 2, margin + room_dims.length + 0.5,
                f'{room_dims.width} м',
                ha='center', va='bottom', fontsize=10, color='red')

        # Длина (справа)
        ax.annotate('',
                    xy=(margin + room_dims.width + 0.3, margin),
                    xytext=(margin + room_dims.width + 0.3, margin + room_dims.length),
                    arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

        ax.text(margin + room_dims.width + 0.5, margin + room_dims.length / 2,
                f'{room_dims.length} м',
                ha='left', va='center', fontsize=10, color='red', rotation=90)

        # Добавляем размерные линии для проемов (окон и дверей)
        self._add_opening_dimensions(ax, room_dims, margin)

    def _add_opening_dimensions(self, ax, room_dims, margin):
        """Добавление размеров для проемов (окон и дверей)."""
        # Эта функция может быть расширена для отображения размеров
        # конкретных проемов, если это необходимо
        pass


def create_simple_floorplan(width, length, windows_count=1, doors_count=1, output_path=None):
    """Создание простой планировки с заданным количеством окон и дверей."""
    from room_detector import RoomDimensions, Window, Door

    room = RoomDimensions(width=width, length=length, height=2.7, area=width * length)

    windows = []
    if windows_count >= 1:
        windows.append(Window(x=width * 0.3, y=1.0, width=1.2, height=1.5, wall='right'))
    if windows_count >= 2:
        windows.append(Window(x=length * 0.4, y=1.0, width=1.0, height=1.2, wall='top'))
    if windows_count >= 3:
        windows.append(Window(x=width * 0.6, y=1.0, width=1.2, height=1.5, wall='left'))

    doors = []
    if doors_count >= 1:
        doors.append(Door(x=width * 0.85, width=0.9, height=2.0, wall='bottom',
                          has_glass=False, is_open=False))
    if doors_count >= 2:
        doors.append(Door(x=length * 0.15, width=0.9, height=2.0, wall='right',
                          has_glass=True, is_open=True))

    drawer = FloorplanDrawer()
    drawer.draw(room, windows, doors, output_path)