import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from PIL import Image
from collections import Counter
import os


@dataclass
class Wall3D:
    """3D стена с параметрами."""
    name: str  # 'left', 'right', 'top', 'bottom'
    start: np.ndarray
    end: np.ndarray
    height: float
    normal: np.ndarray
    windows: List = None
    doors: List = None
    color: Tuple[float, float, float] = (0.85, 0.85, 0.85)

    def __post_init__(self):
        if self.windows is None:
            self.windows = []
        if self.doors is None:
            self.doors = []


class ModelFromFloorplan:
    """Генератор 3D модели на основе 2D планировки."""

    def __init__(self, room_dims, floorplan_image=None):
        self.dims = room_dims
        self.floorplan_image = floorplan_image
        self.walls = []
        self.floor = None

        # Стандартные цвета
        self.wall_color = (0.85, 0.85, 0.85)
        self.floor_color = (0.5, 0.4, 0.3)
        self.door_color = (0.6, 0.4, 0.2)
        self.window_color = (0.7, 0.8, 0.9)

    def extract_floorplan_features(self):
        """Извлечение информации из изображения планировки."""
        if not self.floorplan_image or not Path(self.floorplan_image).exists():
            print("  Изображение планировки не найдено, используем стандартные цвета")
            return

        try:
            img = Image.open(self.floorplan_image)
            img_array = np.array(img)
            h, w = img_array.shape[:2]

            # Анализируем цвета
            wall_samples = []
            floor_samples = []

            # Левый и правый края (стены)
            edge_width = w // 10
            left_edge = img_array[:, :edge_width]
            right_edge = img_array[:, -edge_width:]
            top_edge = img_array[:edge_width, :]

            for edge in [left_edge, right_edge, top_edge]:
                colors = self._get_dominant_colors(edge, n_colors=2)
                wall_samples.extend(colors)

            # Нижний край (пол)
            bottom_height = h // 6
            bottom_edge = img_array[-bottom_height:, :]
            floor_colors = self._get_dominant_colors(bottom_edge, n_colors=2)
            if floor_colors:
                self.floor_color = floor_colors[0]
                print(f"  Цвет пола из планировки: RGB({self.floor_color[0] * 255:.0f}, "
                      f"{self.floor_color[1] * 255:.0f}, {self.floor_color[2] * 255:.0f})")

            if wall_samples:
                self.wall_color = self._average_color(wall_samples)
                print(f"  Цвет стен из планировки: RGB({self.wall_color[0] * 255:.0f}, "
                      f"{self.wall_color[1] * 255:.0f}, {self.wall_color[2] * 255:.0f})")

        except Exception as e:
            print(f"  Ошибка при анализе планировки: {e}")

    def _get_dominant_colors(self, image: np.ndarray, n_colors: int = 3):
        """Получение доминирующих цветов."""
        small = cv2.resize(image, (50, 50))

        if len(small.shape) == 3:
            rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(cv2.cvtColor(small, cv2.COLOR_GRAY2BGR), cv2.COLOR_BGR2RGB)

        pixels = rgb.reshape(-1, 3)
        brightness = np.mean(pixels, axis=1)
        mask = (brightness > 30) & (brightness < 225)
        pixels = pixels[mask]

        if len(pixels) == 0:
            return []

        # Простая группировка цветов
        quantized = (pixels // 30) * 30 + 15
        color_counts = Counter(map(tuple, quantized))

        top_colors = [k for k, v in color_counts.most_common(n_colors)]
        return [tuple(c / 255.0 for c in color) for color in top_colors]

    def _average_color(self, colors: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """Усреднение списка цветов."""
        if not colors:
            return (0.8, 0.8, 0.8)

        avg_r = sum(c[0] for c in colors) / len(colors)
        avg_g = sum(c[1] for c in colors) / len(colors)
        avg_b = sum(c[2] for c in colors) / len(colors)

        return (avg_r, avg_g, avg_b)

    def build_walls(self, windows, doors):
        """Построить 4 стены комнаты с правильной ориентацией."""
        w = self.dims.width
        l = self.dims.length
        h = self.dims.height

        if doors is None:
            doors = []

        print(f"\n  Построение стен комнаты {w}x{l}x{h}")

        # Углы комнаты (X, Y, Z)
        corners = {
            'front_left': np.array([0, 0, 0]),
            'front_right': np.array([w, 0, 0]),
            'back_right': np.array([w, 0, l]),
            'back_left': np.array([0, 0, l])
        }

        # Определяем стены
        walls_config = [
            {
                'name_3d': 'bottom',
                'plan_name': 'top',
                'start': corners['front_left'],
                'end': corners['front_right'],
                'normal': np.array([0, 0, -1]),
                'length': w
            },
            {
                'name_3d': 'right',
                'plan_name': 'right',
                'start': corners['front_right'],
                'end': corners['back_right'],
                'normal': np.array([1, 0, 0]),
                'length': l
            },
            {
                'name_3d': 'top',
                'plan_name': 'bottom',
                'start': corners['back_right'],
                'end': corners['back_left'],
                'normal': np.array([0, 0, 1]),
                'length': w
            },
            {
                'name_3d': 'left',
                'plan_name': 'left',
                'start': corners['back_left'],
                'end': corners['front_left'],
                'normal': np.array([-1, 0, 0]),
                'length': l
            }
        ]

        # Для каждой стены собираем окна и двери
        for wall_conf in walls_config:
            wall_name_3d = wall_conf['name_3d']
            plan_name = wall_conf['plan_name']
            wall_length = wall_conf['length']

            # Собираем окна
            wall_windows = []
            for win in windows:
                if win.wall == plan_name:
                    x_pos = max(0.1, min(win.x, wall_length - win.width - 0.1))
                    from room_detector import Window
                    wall_windows.append(Window(
                        x=x_pos,
                        y=win.y,
                        width=win.width,
                        height=win.height,
                        wall=wall_name_3d
                    ))
                    print(f"    Окно на стене {plan_name} -> {wall_name_3d}: x={x_pos:.2f}")

            # Собираем двери
            wall_doors = []
            for door in doors:
                if door.wall == plan_name:
                    x_pos = max(0.1, min(door.x, wall_length - door.width - 0.1))
                    from room_detector import Door
                    wall_doors.append(Door(
                        x=x_pos,
                        width=door.width,
                        height=door.height,
                        wall=wall_name_3d,
                        has_glass=getattr(door, 'has_glass', False),
                        is_open=getattr(door, 'is_open', False)
                    ))
                    print(f"    Дверь на стене {plan_name} -> {wall_name_3d}: x={x_pos:.2f}")

            # Создаем стену
            wall = Wall3D(
                name=wall_name_3d,
                start=wall_conf['start'],
                end=wall_conf['end'],
                height=h,
                normal=wall_conf['normal'],
                windows=wall_windows,
                doors=wall_doors,
                color=self.wall_color
            )
            self.walls.append(wall)
            print(f"  Стена {wall_name_3d}: длина={wall_length:.2f}м, "
                  f"окон={len(wall_windows)}, дверей={len(wall_doors)}")

        # Пол
        self.floor = {
            'corners': [
                corners['front_left'],
                corners['front_right'],
                corners['back_right'],
                corners['back_left']
            ],
            'normal': np.array([0, 1, 0]),
            'color': self.floor_color
        }


    def _build_wall_segments(self, wall, vertex_offset, wall_idx):
        """Построение стены с разбиением на сегменты."""
        vertices = []
        faces = []

        start, end = wall.start, wall.end
        height = wall.height
        wall_vec = end - start
        wall_len = np.linalg.norm(wall_vec)

        if wall_len < 0.01:
            return vertices, faces, vertex_offset

        wall_dir = wall_vec / wall_len

        # Собираем все проемы
        all_openings = []

        for win in wall.windows:
            all_openings.append({
                'x': win.x,
                'y': win.y,
                'width': win.width,
                'height': win.height,
                'type': 'window'
            })

        for door in wall.doors:
            all_openings.append({
                'x': door.x,
                'y': 0,
                'width': door.width,
                'height': door.height,
                'type': 'door'
            })

        all_openings.sort(key=lambda op: op['x'])

        # Точки разбиения по X
        x_points = [0.0]
        for op in all_openings:
            x1 = max(op['x'], 0.0)
            x2 = min(op['x'] + op['width'], wall_len)
            if x1 > x_points[-1] + 0.05:
                x_points.append(x1)
            if x2 > x_points[-1] + 0.05 and x2 < wall_len - 0.05:
                x_points.append(x2)

        if x_points[-1] < wall_len - 0.05:
            x_points.append(wall_len)

        # Для каждого сегмента
        for i in range(len(x_points) - 1):
            x_start, x_end = x_points[i], x_points[i + 1]

            seg_openings = [op for op in all_openings
                            if not (op['x'] + op['width'] <= x_start or op['x'] >= x_end)]

            # Точки разбиения по Y
            y_points = [0.0]
            for op in seg_openings:
                y1 = op['y']
                y2 = op['y'] + op['height']
                if y1 > y_points[-1] + 0.05:
                    y_points.append(y1)
                if y2 > y_points[-1] + 0.05 and y2 < height - 0.05:
                    y_points.append(y2)

            if y_points[-1] < height - 0.05:
                y_points.append(height)

            for j in range(len(y_points) - 1):
                y_start, y_end = y_points[j], y_points[j + 1]

                # Проверяем, проем ли это
                is_opening = False
                opening_type = 'wall'

                for op in seg_openings:
                    if (op['y'] - 0.05 <= y_start <= op['y'] + op['height'] + 0.05 and
                            op['x'] - 0.05 <= x_start <= op['x'] + op['width'] + 0.05):
                        is_opening = True
                        opening_type = op['type']
                        break

                # Вершины
                p0 = start + wall_dir * x_start + np.array([0, y_start, 0])
                p1 = start + wall_dir * x_end + np.array([0, y_start, 0])
                p2 = start + wall_dir * x_end + np.array([0, y_end, 0])
                p3 = start + wall_dir * x_start + np.array([0, y_end, 0])

                base = len(vertices)
                vertices.extend([p0, p1, p2, p3])

                # Материал
                if is_opening:
                    if opening_type == 'door':
                        mat = f"door_{wall_idx}"
                    else:
                        mat = f"window_{wall_idx}"
                else:
                    mat = f"wall_{wall_idx}"

                # Грани (два треугольника)
                faces.extend([
                    (mat, base + vertex_offset, base + 1 + vertex_offset, base + 2 + vertex_offset),
                    (mat, base + vertex_offset, base + 2 + vertex_offset, base + 3 + vertex_offset)
                ])

        return vertices, faces, vertex_offset + len(vertices)

    def _build_plane(self, corners, vertex_offset, color_name):
        """Построение плоскости (пол или потолок)."""
        if len(corners) < 4:
            return [], [], vertex_offset

        v0, v1, v2, v3 = corners[:4]
        base = len(vertices) if 'vertices' in locals() else 0

        vertices = [v0, v1, v2, v3]

        # Грани (два треугольника)
        faces = [
            (color_name, base + vertex_offset, base + 1 + vertex_offset, base + 2 + vertex_offset),
            (color_name, base + vertex_offset, base + 2 + vertex_offset, base + 3 + vertex_offset)
        ]

        return vertices, faces, vertex_offset + len(vertices)

    def export_obj(self, filename):
        """Экспорт в Wavefront OBJ формат."""
        all_vertices = []
        all_faces = []
        vertex_offset = 1

        mtl_filename = filename.replace('.obj', '.mtl')

        # Собираем все вершины и грани
        print("\n  Генерация геометрии:")

        # Стены
        for i, wall in enumerate(self.walls):
            verts, faces, vertex_offset = self._build_wall_segments(wall, vertex_offset, i)
            all_vertices.extend(verts)
            all_faces.extend(faces)
            print(f"    Стена {wall.name}: {len(verts)} вершин, {len(faces)} граней")

        # Пол
        verts, faces, vertex_offset = self._build_plane(
            self.floor['corners'],
            vertex_offset,
            "floor"
        )
        all_vertices.extend(verts)
        all_faces.extend(faces)
        print(f"    Пол: {len(verts)} вершин, {len(faces)} граней")

        # Записываем OBJ файл
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(f"# Room 3D Model - Generated from floorplan\n")
            f.write(f"mtllib {Path(mtl_filename).name}\n\n")

            # Вершины
            for v in all_vertices:
                f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")

            # Текстурные координаты
            f.write("\nvt 0.0 0.0\nvt 1.0 0.0\nvt 1.0 1.0\nvt 0.0 1.0\n")

            # Нормали
            f.write("\n")
            for wall in self.walls:
                f.write(f"vn {wall.normal[0]:.4f} {wall.normal[1]:.4f} {wall.normal[2]:.4f}\n")
            f.write(f"vn {self.floor['normal'][0]:.4f} {self.floor['normal'][1]:.4f} {self.floor['normal'][2]:.4f}\n")


            # Грани
            f.write("\n")
            current_mtl = None
            wall_normal_offset = 1
            floor_normal_offset = wall_normal_offset + len(self.walls)
            ceiling_normal_offset = floor_normal_offset + 1

            for face in all_faces:
                mat = face[0]
                if mat != current_mtl:
                    current_mtl = mat
                    f.write(f"usemtl {mat}\n")

                # Определяем индекс нормали
                if mat == "floor":
                    ni = floor_normal_offset
                elif mat == "ceiling":
                    ni = ceiling_normal_offset
                else:
                    wall_idx = int(mat.split('_')[1]) if '_' in mat and len(mat.split('_')) > 1 else 0
                    ni = wall_normal_offset + wall_idx

                if len(face) == 4:  # Треугольник
                    _, v1, v2, v3 = face
                    f.write(f"f {v1}/{1}/{ni} {v2}/{2}/{ni} {v3}/{3}/{ni}\n")
                elif len(face) == 5:  # Четырехугольник
                    _, v1, v2, v3, v4 = face
                    f.write(f"f {v1}/{1}/{ni} {v2}/{2}/{ni} {v3}/{3}/{ni} {v4}/{4}/{ni}\n")

        # Создаем MTL файл
        self._write_mtl(mtl_filename)

        print(f"\n✓ 3D модель сохранена: {filename}")
        print(f"  Вершин: {len(all_vertices)}")
        print(f"  Граней: {len(all_faces)}")

    def _write_mtl(self, filename):
        """Запись MTL файла с материалами."""
        with open(filename, 'w', encoding='utf-8') as f:
            f.write("# Room materials - Generated from floorplan\n")

            # Материалы для стен
            for i, wall in enumerate(self.walls):
                r, g, b = wall.color
                f.write(f"""
newmtl wall_{i}
Ka {r:.3f} {g:.3f} {b:.3f}
Kd {r:.3f} {g:.3f} {b:.3f}
Ks 0.1 0.1 0.1
Ns 10
illum 2
""")

                wr, wg, wb = self.window_color
                f.write(f"""
newmtl window_{i}
Ka {wr:.3f} {wg:.3f} {wb:.3f}
Kd {wr:.3f} {wg:.3f} {wb:.3f}
Ks 0.9 0.9 0.95
Ns 100
d 0.6
illum 4
""")

                dr, dg, db = self.door_color
                f.write(f"""
newmtl door_{i}
Ka {dr:.3f} {dg:.3f} {db:.3f}
Kd {dr:.3f} {dg:.3f} {db:.3f}
Ks 0.2 0.2 0.2
Ns 20
illum 2
""")

            # Пол
            fr, fg, fb = self.floor_color
            f.write(f"""
newmtl floor
Ka {fr:.3f} {fg:.3f} {fb:.3f}
Kd {fr:.3f} {fg:.3f} {fb:.3f}
Ks 0.1 0.1 0.1
Ns 5
illum 2
""")


def create_3d_model_from_floorplan(room_dims, windows, doors, output_path, floorplan_path=None):
    """Создание 3D модели на основе 2D планировки."""
    print("ПОСТРОЕНИЕ 3D МОДЕЛИ НА ОСНОВЕ ПЛАНИРОВКИ")
    print(f"Размеры комнаты: {room_dims.width}м x {room_dims.length}м x {room_dims.height}м")
    print(f"Окон: {len(windows)}")
    for w in windows:
        print(f"  Окно: стена={w.wall}, x={w.x:.2f}м, {w.width}x{w.height}м")
    print(f"Дверей: {len(doors)}")
    for d in doors:
        print(f"  Дверь: стена={d.wall}, x={d.x:.2f}м, {d.width}x{d.height}м")
    print("=" * 60)

    model = ModelFromFloorplan(room_dims, floorplan_path)
    model.extract_floorplan_features()
    model.build_walls(windows, doors)
    model.export_obj(output_path)

    return model