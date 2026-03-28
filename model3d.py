import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from collections import Counter
from room_detector import RoomDimensions, Window, Door


@dataclass
class Wall3D:
    """3D стена с параметрами."""
    start: np.ndarray
    end: np.ndarray
    height: float
    normal: np.ndarray
    windows: List[Window] = None
    doors: List[Door] = None
    color: Tuple[float, float, float] = (0.9, 0.9, 0.9)

    def __post_init__(self):
        if self.windows is None:
            self.windows = []
        if self.doors is None:
            self.doors = []


class RoomModel3D:
    """Генератор 3D модели комнаты."""

    def __init__(self, room_dims: RoomDimensions):
        self.dims = room_dims
        self.walls = []
        self.floor = None
        self.floor_color = (0.5, 0.4, 0.3)
        self.ceiling = None
        self.ceiling_color = (0.95, 0.95, 0.95)

    def analyze_photo_colors(self, images: List[np.ndarray], windows_data: List[Dict] = None):
        """
        Анализ цветов с фотографий - выбирает самый частый цвет для каждой стены и пола.
        """
        print("\n  Анализ цветов с фотографий...")

        if not images:
            print("  Нет фотографий для анализа цветов, используем стандартные цвета")
            return

        wall_samples = {'left': [], 'right': [], 'top': [], 'bottom': []}
        floor_samples = []
        ceiling_samples = []

        for i, img in enumerate(images):
            if img is None:
                continue

            print(f"    Обработка фото {i + 1}...")
            h, w = img.shape[:2]

            # Разбиваем на зоны
            zones = {
                'left': img[:, :w // 4],
                'center': img[:, w // 4:3 * w // 4],
                'right': img[:, 3 * w // 4:],
                'floor': img[2 * h // 3:, :],
                'ceiling': img[:h // 3, :]
            }

            # Анализ каждой зоны
            for zone_name, zone_img in zones.items():
                if zone_img.size == 0:
                    continue

                # Получаем доминирующий цвет в зоне
                dominant_color = self._get_dominant_color(zone_img)

                if zone_name == 'floor':
                    floor_samples.append(dominant_color)
                elif zone_name == 'ceiling':
                    ceiling_samples.append(dominant_color)
                else:
                    # Определяем стену
                    wall_map = {'left': 'left', 'center': 'top', 'right': 'right'}
                    wall_key = wall_map.get(zone_name)
                    if wall_key:
                        wall_samples[wall_key].append(dominant_color)

        # Выбираем цвета для стен
        for wall_name, colors in wall_samples.items():
            if colors:
                color = self._most_common_color(colors)
                print(f"    Стена {wall_name}: RGB({color[0] * 255:.0f}, {color[1] * 255:.0f}, {color[2] * 255:.0f})")
                # Применяем к стенам
                for wall in self.walls:
                    if self._get_wall_name(wall) == wall_name:
                        wall.color = color

        # Пол
        if floor_samples:
            self.floor_color = self._most_common_color(floor_samples)
            print(
                f"    Пол: RGB({self.floor_color[0] * 255:.0f}, {self.floor_color[1] * 255:.0f}, {self.floor_color[2] * 255:.0f})")

        # Потолок
        if ceiling_samples:
            self.ceiling_color = self._most_common_color(ceiling_samples)
            print(
                f"    Потолок: RGB({self.ceiling_color[0] * 255:.0f}, {self.ceiling_color[1] * 255:.0f}, {self.ceiling_color[2] * 255:.0f})")

    def _get_dominant_color(self, image: np.ndarray) -> Tuple[float, float, float]:
        """
        Находит доминирующий цвет в изображении.
        """
        # Уменьшаем для скорости
        small = cv2.resize(image, (30, 30))

        # Конвертируем в RGB
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

        # Фильтруем очень темные и очень светлые пиксели
        pixels = rgb.reshape(-1, 3)
        brightness = np.mean(pixels, axis=1)
        valid_pixels = pixels[(brightness > 40) & (brightness < 215)]

        if len(valid_pixels) == 0:
            return (0.8, 0.8, 0.8)

        # Группируем похожие цвета
        quantized = (valid_pixels // 25) * 25 + 12

        # Считаем частоту
        color_counts = Counter(map(tuple, quantized))
        most_common = color_counts.most_common(1)[0][0]

        # Нормализуем в [0, 1]
        return tuple(c / 255.0 for c in most_common)

    def _most_common_color(self, colors: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Выбирает самый частый цвет из списка.
        """
        if not colors:
            return (0.8, 0.8, 0.8)

        # Округляем для группировки похожих
        quantized = [(round(r, 1), round(g, 1), round(b, 1)) for r, g, b in colors]

        color_counts = Counter(quantized)
        most_common = color_counts.most_common(1)[0][0]

        return most_common

    def _get_wall_name(self, wall: Wall3D) -> str:
        """Определить имя стены по нормали."""
        normals = {
            'bottom': np.array([0, 0, -1]),
            'right': np.array([1, 0, 0]),
            'top': np.array([0, 0, 1]),
            'left': np.array([-1, 0, 0])
        }
        for name, n in normals.items():
            if np.allclose(wall.normal, n, atol=1e-3):
                return name
        return 'unknown'

    def build_walls(self, windows: List[Window], doors: List[Door] = None):
        """Построить 4 стены комнаты с вырезами под окна и двери."""
        w, l, h = self.dims.width, self.dims.length, self.dims.height

        if doors is None:
            doors = []

        windows_by_wall = {'left': [], 'right': [], 'top': [], 'bottom': []}
        for win in windows:
            if win.wall in windows_by_wall:
                windows_by_wall[win.wall].append(win)

        doors_by_wall = {'left': [], 'right': [], 'top': [], 'bottom': []}
        for door in doors:
            if door.wall in doors_by_wall:
                doors_by_wall[door.wall].append(door)

        # Углы комнаты (X, Y, Z) где Y - высота
        corners = [
            np.array([0, 0, 0]),  # левый нижний угол
            np.array([w, 0, 0]),  # правый нижний угол
            np.array([w, 0, l]),  # правый верхний угол
            np.array([0, 0, l])  # левый верхний угол
        ]

        wall_configs = [
            (
            'bottom', corners[0], corners[1], np.array([0, 0, -1]), windows_by_wall['bottom'], doors_by_wall['bottom']),
            ('right', corners[1], corners[2], np.array([1, 0, 0]), windows_by_wall['right'], doors_by_wall['right']),
            ('top', corners[2], corners[3], np.array([0, 0, 1]), windows_by_wall['top'], doors_by_wall['top']),
            ('left', corners[3], corners[0], np.array([-1, 0, 0]), windows_by_wall['left'], doors_by_wall['left']),
        ]

        for name, start, end, normal, wall_windows, wall_doors in wall_configs:
            wall = Wall3D(start=start, end=end, height=h, normal=normal,
                          windows=wall_windows, doors=wall_doors)
            self.walls.append(wall)
            print(
                f"  Стена {name}: {np.linalg.norm(end - start):.2f}м, окон: {len(wall_windows)}, дверей: {len(wall_doors)}")

        self.floor = {
            'corners': corners,
            'normal': np.array([0, 1, 0])
        }

        self.ceiling = {
            'corners': [c + np.array([0, h, 0]) for c in corners],
            'normal': np.array([0, -1, 0])
        }

    def export_obj(self, filename: str, images: List[np.ndarray] = None, windows_data: List[Dict] = None):
        """Экспорт в Wavefront OBJ с цветами из фото."""

        if images and len(images) > 0:
            self.analyze_photo_colors(images, windows_data)

        all_vertices = []
        all_faces = []
        vertex_offset = 1

        mtl_filename = filename.replace('.obj', '.mtl')

        lines = [
            "# Room 3D Model - Generated by RoomPlanner",
            f"mtllib {Path(mtl_filename).name}",
            ""
        ]

        # Генерируем стены
        for i, wall in enumerate(self.walls):
            lines.append(f"# Wall {i + 1} ({self._get_wall_name(wall)})")
            verts, faces, vertex_offset = self._build_wall_with_holes(wall, vertex_offset, i)
            all_vertices.extend(verts)
            all_faces.extend(faces)

        # Генерируем пол
        lines.append("# Floor")
        verts, faces, vertex_offset = self._build_floor(self.floor['corners'], vertex_offset)
        all_vertices.extend(verts)
        all_faces.extend([("floor", *f) for f in faces])

        # Генерируем потолок
        lines.append("# Ceiling")
        verts, faces, vertex_offset = self._build_ceiling(self.ceiling['corners'], vertex_offset)
        all_vertices.extend(verts)
        all_faces.extend([("ceiling", *f) for f in faces])

        # Записываем вершины
        for v in all_vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        # Текстурные координаты
        lines.extend(["", "vt 0.0 0.0", "vt 1.0 0.0", "vt 1.0 1.0", "vt 0.0 1.0"])

        # Нормали
        lines.append("")
        for wall in self.walls:
            lines.append(f"vn {wall.normal[0]:.4f} {wall.normal[1]:.4f} {wall.normal[2]:.4f}")
        lines.append(f"vn {self.floor['normal'][0]:.4f} {self.floor['normal'][1]:.4f} {self.floor['normal'][2]:.4f}")
        lines.append(
            f"vn {self.ceiling['normal'][0]:.4f} {self.ceiling['normal'][1]:.4f} {self.ceiling['normal'][2]:.4f}")

        # Записываем грани
        lines.append("")
        current_mtl = None
        wall_normal_offset = 1
        floor_normal_offset = wall_normal_offset + len(self.walls)
        ceiling_normal_offset = floor_normal_offset + 1

        for face in all_faces:
            mat = face[0]
            if mat != current_mtl:
                current_mtl = mat
                lines.append(f"usemtl {mat}")

            # Определяем индекс нормали
            if mat == "floor":
                ni = floor_normal_offset
            elif mat == "ceiling":
                ni = ceiling_normal_offset
            else:
                # Для стен используем соответствующий индекс
                wall_idx = int(mat.split('_')[1]) if '_' in mat else 0
                ni = wall_normal_offset + wall_idx

            # Формируем строку грани (используем 4 вершины для квадратов)
            if len(face) == 5:  # Треугольник
                _, v1, v2, v3, _ = face
                lines.append(f"f {v1}/{1}/{ni} {v2}/{2}/{ni} {v3}/{3}/{ni}")
            elif len(face) == 6:  # Четырехугольник
                _, v1, v2, v3, v4, _ = face
                lines.append(f"f {v1}/{1}/{ni} {v2}/{2}/{ni} {v3}/{3}/{ni} {v4}/{4}/{ni}")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))

        self._write_mtl_colored(mtl_filename)

        print(f"\n✓ 3D модель сохранена: {filename}")
        print(f"  Вершин: {len(all_vertices)}")
        print(f"  Граней: {len(all_faces)}")
        print(f"  Материалы: {mtl_filename}")

    def _build_wall_with_holes(self, wall: Wall3D, vertex_offset: int, wall_idx: int):
        """Построить стену с вырезами под окна и двери."""
        vertices = []
        faces = []

        start, end = wall.start, wall.end
        height = wall.height
        wall_vec = end - start
        wall_len = np.linalg.norm(wall_vec)

        if wall_len < 0.01:
            return vertices, faces, vertex_offset

        wall_dir = wall_vec / wall_len

        wall_normal_idx = wall_idx

        # Собираем все проемы (окна + двери)
        all_openings = []

        # Добавляем окна
        for win in wall.windows:
            all_openings.append({
                'x': win.x,
                'y': win.y,
                'width': win.width,
                'height': win.height,
                'type': 'window'
            })

        # Добавляем двери
        for door in wall.doors:
            all_openings.append({
                'x': door.x,
                'y': 0,  # Дверь от пола
                'width': door.width,
                'height': door.height,
                'type': 'door'
            })

        # Сортируем проемы по X
        all_openings.sort(key=lambda op: op['x'])

        # Создаем позиции по X для разбиения стены
        x_positions = [0.0]

        for op in all_openings:
            x1 = max(op['x'], 0.0)
            x2 = min(op['x'] + op['width'], wall_len)

            if x1 > x_positions[-1] + 0.05:
                x_positions.append(x1)
            if x2 > x_positions[-1] + 0.05 and x2 < wall_len - 0.05:
                x_positions.append(x2)

        if x_positions[-1] < wall_len - 0.05:
            x_positions.append(wall_len)

        # Для каждого сегмента по горизонтали
        for col_idx in range(len(x_positions) - 1):
            x_start, x_end = x_positions[col_idx], x_positions[col_idx + 1]

            # Находим проемы в этом сегменте
            col_openings = [op for op in all_openings
                            if not (op['x'] + op['width'] <= x_start or op['x'] >= x_end)]

            # Создаем позиции по Y для разбиения
            y_positions = [0.0]
            for op in col_openings:
                y1 = op['y']
                y2 = op['y'] + op['height']
                if y1 > y_positions[-1] + 0.05:
                    y_positions.append(y1)
                if y2 > y_positions[-1] + 0.05 and y2 < height - 0.05:
                    y_positions.append(y2)

            if y_positions[-1] < height - 0.05:
                y_positions.append(height)

            # Для каждого сегмента по вертикали
            for row_idx in range(len(y_positions) - 1):
                y1, y2 = y_positions[row_idx], y_positions[row_idx + 1]

                # Проверяем, является ли этот сегмент проемом
                is_opening = False
                opening_type = 'wall'

                for op in col_openings:
                    if (op['y'] - 0.05 <= y1 <= op['y'] + op['height'] + 0.05 and
                            op['x'] - 0.05 <= x_start <= op['x'] + op['width'] + 0.05):
                        is_opening = True
                        opening_type = op['type']
                        break

                # Вычисляем вершины сегмента
                p0 = start + wall_dir * x_start + np.array([0, y1, 0])
                p1 = start + wall_dir * x_end + np.array([0, y1, 0])
                p2 = start + wall_dir * x_end + np.array([0, y2, 0])
                p3 = start + wall_dir * x_start + np.array([0, y2, 0])

                base = vertex_offset + len(vertices)
                vertices.extend([p0, p1, p2, p3])

                # Определяем материал
                if is_opening:
                    if opening_type == 'door':
                        mat = f"door_{wall_idx}"
                    else:
                        mat = f"window_{wall_idx}"
                else:
                    mat = f"wall_{wall_idx}"

                # Добавляем грани (два треугольника для прямоугольника)
                faces.extend([
                    (mat, base, base + 1, base + 2, wall_idx),
                    (mat, base, base + 2, base + 3, wall_idx)
                ])

        return vertices, faces, vertex_offset + len(vertices)

    def _build_floor(self, corners: List[np.ndarray], vertex_offset: int):
        """Построить пол."""
        v0, v1, v2, v3 = corners
        base = vertex_offset

        vertices = [v0, v1, v2, v3]
        faces = [
            (base, base + 1, base + 2),
            (base, base + 2, base + 3)
        ]

        return vertices, faces, vertex_offset + 4

    def _build_ceiling(self, corners: List[np.ndarray], vertex_offset: int):
        """Построить потолок."""
        v0, v1, v2, v3 = corners
        base = vertex_offset

        vertices = [v0, v1, v2, v3]
        faces = [
            (base, base + 1, base + 2),
            (base, base + 2, base + 3)
        ]

        return vertices, faces, vertex_offset + 4

    def _write_mtl_colored(self, filename: str):
        """Записать MTL с реальными цветами."""
        lines = ["# Room materials - Generated by RoomPlanner"]

        # Материалы для стен
        for i, wall in enumerate(self.walls):
            r, g, b = wall.color

            # Основной цвет стены
            lines.append(f"""
newmtl wall_{i}
Ka {r:.3f} {g:.3f} {b:.3f}
Kd {r:.3f} {g:.3f} {b:.3f}
Ks 0.1 0.1 0.1
Ns 10
illum 2""")

            # Окно — светлее и полупрозрачное
            wr = min(r + 0.2, 1.0)
            wg = min(g + 0.2, 1.0)
            wb = min(b + 0.3, 1.0)
            lines.append(f"""
newmtl window_{i}
Ka {wr:.3f} {wg:.3f} {wb:.3f}
Kd {wr:.3f} {wg:.3f} {wb:.3f}
Ks 0.9 0.9 0.95
Ns 100
d 0.6
illum 4""")

            # Дверь — коричневый цвет дерева
            lines.append(f"""
newmtl door_{i}
Ka 0.4 0.25 0.1
Kd 0.5 0.35 0.15
Ks 0.2 0.2 0.2
Ns 20
illum 2""")

        # Пол
        fr, fg, fb = self.floor_color
        lines.append(f"""
newmtl floor
Ka {fr:.3f} {fg:.3f} {fb:.3f}
Kd {fr:.3f} {fg:.3f} {fb:.3f}
Ks 0.1 0.1 0.1
Ns 5
illum 2""")

        # Потолок
        cr, cg, cb = self.ceiling_color
        lines.append(f"""
newmtl ceiling
Ka {cr:.3f} {cg:.3f} {cb:.3f}
Kd {cr:.3f} {cg:.3f} {cb:.3f}
Ks 0.1 0.1 0.1
Ns 5
illum 2""")

        with open(filename, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))


def create_3d_model(room_dims: RoomDimensions, windows: List[Window], doors: List[Door],
                    output_path: str, images: List[np.ndarray] = None,
                    windows_data: List[Dict] = None):
    """Создать 3D модель с цветами из фото и дверями."""
    print("\nПОСТРОЕНИЕ 3D МОДЕЛИ")
    print(f"Размеры: {room_dims.width}м x {room_dims.length}м x {room_dims.height}м")
    print(f"Окон: {len(windows)}, Дверей: {len(doors)}")

    model = RoomModel3D(room_dims)
    model.build_walls(windows, doors)
    model.export_obj(output_path, images=images, windows_data=windows_data)

    return model