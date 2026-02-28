import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Rectangle, FancyBboxPatch
import numpy as np


class FloorplanDrawer:
    """Рисование планировки комнаты."""

    def __init__(self, pixels_per_meter=100):
        self.pixels_per_meter = pixels_per_meter

    def draw(self, room_dims, windows, output_path=None):
        # Размеры изображения
        margin = 1.0  # метры
        total_width = room_dims.width + 2 * margin
        total_length = room_dims.length + 2 * margin

        fig_width = total_width * self.pixels_per_meter / 100
        fig_height = total_length * self.pixels_per_meter / 100

        fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

        # Рисуем комнату (стены)
        room_rect = Rectangle(
            (margin, margin),
            room_dims.width,
            room_dims.length,
            linewidth=3,
            edgecolor='black',
            facecolor='lightgray',
            alpha=0.3
        )
        ax.add_patch(room_rect)

        # Рисуем окна
        for window in windows:
            self._draw_window(ax, window, margin, room_dims)

        # Добавляем размеры
        self._add_dimensions(ax, room_dims, margin)

        # Настройка осей
        ax.set_xlim(0, total_width)
        ax.set_ylim(0, total_length)
        ax.set_aspect('equal')
        ax.axis('off')

        # Заголовок
        plt.title(f"Планировка комнаты\n"
                 f"{room_dims.width}м × {room_dims.length}м = {room_dims.area}м²",
                 fontsize=14, fontweight='bold')

        plt.tight_layout()

        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight',
                       facecolor='white', edgecolor='none')
            print(f"\nПланировка сохранена: {output_path}")
        else:
            plt.show()

        plt.close()

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

    def _add_dimensions(self, ax, room_dims, margin):
        """Добавление размеров на чертеж."""
        # Ширина (сверху)
        ax.annotate('',
                   xy=(margin + room_dims.width, margin + room_dims.length + 0.3),
                   xytext=(margin, margin + room_dims.length + 0.3),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

        ax.text(margin + room_dims.width/2, margin + room_dims.length + 0.5,
               f'{room_dims.width} м',
               ha='center', va='bottom', fontsize=10, color='red')

        # Длина (справа)
        ax.annotate('',
                   xy=(margin + room_dims.width + 0.3, margin),
                   xytext=(margin + room_dims.width + 0.3, margin + room_dims.length),
                   arrowprops=dict(arrowstyle='<->', color='red', lw=1.5))

        ax.text(margin + room_dims.width + 0.5, margin + room_dims.length/2,
               f'{room_dims.length} м',
               ha='left', va='center', fontsize=10, color='red', rotation=90)


def create_simple_floorplan(width, length, windows_count=1, output_path=None):
    from room_detector import RoomDimensions, Window

    room = RoomDimensions(width=width, length=length, height=2.7, area=width*length)

    windows = []
    if windows_count >= 1:
        windows.append(Window(x=width*0.3, y=1.0, width=1.2, height=1.5, wall='right'))
    if windows_count >= 2:
        windows.append(Window(x=length*0.4, y=1.0, width=1.0, height=1.2, wall='top'))

    drawer = FloorplanDrawer()
    drawer.draw(room, windows, output_path)