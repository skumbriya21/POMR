import numpy as np
from typing import List
from dataclasses import dataclass
from pathlib import Path
from room_detector import RoomDimensions, Window


@dataclass
class Wall3D:
    """3D стена с параметрами."""
    start: np.ndarray
    end: np.ndarray
    height: float
    normal: np.ndarray
    windows: List[Window] = None

    def __post_init__(self):
        if self.windows is None:
            self.windows = []


class RoomModel3D:
    """Генератор 3D модели комнаты."""

    def __init__(self, room_dims: RoomDimensions):
        self.dims = room_dims
        self.walls = []
        self.floor = None

    def build_walls(self, windows: List[Window]):
        """Построить 4 стены комнаты с вырезами под окна."""
        w, l, h = self.dims.width, self.dims.length, self.dims.height

        # Распределяем окна по стенам
        windows_by_wall = {'left': [], 'right': [], 'top': [], 'bottom': []}
        for win in windows:
            if win.wall in windows_by_wall:
                windows_by_wall[win.wall].append(win)

        # Координаты углов (Y вверх)
        corners = [
            np.array([0, 0, 0]),      # 0: bottom-left
            np.array([w, 0, 0]),      # 1: bottom-right
            np.array([w, 0, l]),      # 2: top-right
            np.array([0, 0, l])       # 3: top-left
        ]

        # Стены: bottom, right, top, left
        wall_configs = [
            ('bottom', corners[0], corners[1], np.array([0, 0, -1]), windows_by_wall['bottom']),
            ('right', corners[1], corners[2], np.array([1, 0, 0]), windows_by_wall['right']),
            ('top', corners[2], corners[3], np.array([0, 0, 1]), windows_by_wall['top']),
            ('left', corners[3], corners[0], np.array([-1, 0, 0]), windows_by_wall['left']),
        ]

        for name, start, end, normal, wall_windows in wall_configs:
            wall = Wall3D(start=start, end=end, height=h, normal=normal, windows=wall_windows)
            self.walls.append(wall)
            print(f"  Стена {name}: {np.linalg.norm(end-start):.2f}м, окон: {len(wall_windows)}")

        # Только пол (без потолка)
        self.floor = {
            'corners': [corners[0], corners[1], corners[2], corners[3]],
            'normal': np.array([0, 1, 0])
        }

    def export_obj(self, filename: str):
        """Экспорт в Wavefront OBJ с вырезами под окна."""
        all_vertices = []
        all_faces = []  # (material, v1, v2, v3, normal_idx)
        vertex_offset = 1

        mtl_filename = filename.replace('.obj', '.mtl')

        lines = [
            "# Room 3D Model - Walls with window holes",
            f"mtllib {Path(mtl_filename).name}",
            ""
        ]

        # Стены с вырезами
        for i, wall in enumerate(self.walls):
            lines.append(f"# Wall {i+1}")
            verts, faces, vertex_offset = self._build_wall_with_holes(wall, vertex_offset)
            all_vertices.extend(verts)
            all_faces.extend(faces)

        # Пол
        lines.append("# Floor")
        verts, faces, vertex_offset = self._build_floor(self.floor['corners'], vertex_offset)
        all_vertices.extend(verts)
        all_faces.extend([("floor", *f) for f in faces])

        # Записываем вершины
        for v in all_vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        # Текстурные координаты
        lines.extend(["", "vt 0.0 0.0", "vt 1.0 0.0", "vt 1.0 1.0", "vt 0.0 1.0"])

        # Нормали (по порядку: стены, пол)
        lines.append("")
        for wall in self.walls:
            lines.append(f"vn {wall.normal[0]:.4f} {wall.normal[1]:.4f} {wall.normal[2]:.4f}")
        lines.append("vn 0.0 1.0 0.0")  # пол

        # Грани
        lines.append("")
        current_mtl = None
        floor_ni = len(self.walls) + 1

        for face in all_faces:
            mat = face[0]
            if mat != current_mtl:
                current_mtl = mat
                lines.append(f"usemtl {mat}")

            # Определяем нормаль
            if mat == "floor":
                ni = floor_ni
            else:
                # Находим индекс стены по нормали
                ni = 1  # default
                for i, w in enumerate(self.walls, 1):
                    if face[0] in ['wall', 'window'] and len(face) > 1:
                        # Для стен используем их нормаль
                        ni = i
                        break

            # face: (mat, v1, v2, v3, ni) или (mat, v1, v2, v3, v4, ni)
            if len(face) == 5:  # треугольник
                _, v1, v2, v3, ni_face = face
                lines.append(f"f {v1}/{1}/{ni_face} {v2}/{2}/{ni_face} {v3}/{3}/{ni_face}")
            elif len(face) == 6:  # четырёхугольник
                _, v1, v2, v3, v4, ni_face = face
                lines.append(f"f {v1}/{1}/{ni_face} {v2}/{2}/{ni_face} {v3}/{3}/{ni_face} {v4}/{4}/{ni_face}")

        # Сохраняем
        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self._write_mtl(mtl_filename)

        print(f"\n✓ 3D модель сохранена: {filename}")
        print(f"  Вершин: {len(all_vertices)}")
        print(f"  Граней: {len(all_faces)}")

    def _build_wall_with_holes(self, wall: Wall3D, vertex_offset: int):
        """
        Построить стену с настоящими вырезами под окна.
        Стена разбивается на части: до окна, между окнами, после окна.
        """
        vertices = []
        faces = []

        start, end = wall.start, wall.end
        height = wall.height
        wall_vec = end - start
        wall_len = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_len if wall_len > 0 else np.array([1, 0, 0])

        # Нормаль стены для всех её граней
        wall_normal_idx = len(self.walls)  # будет определено позже, пока используем индекс стены

        if not wall.windows:
            # Сплошная стена без окон
            v0 = start
            v1 = end
            v2 = end + np.array([0, height, 0])
            v3 = start + np.array([0, height, 0])

            base = vertex_offset
            vertices = [v0, v1, v2, v3]
            # Два треугольника для четырёхугольника
            faces = [
                ("wall", base, base+1, base+2, wall_normal_idx),
                ("wall", base, base+2, base+3, wall_normal_idx)
            ]
            return vertices, faces, vertex_offset + 4

        # Сортируем окна по позиции
        sorted_wins = sorted(wall.windows, key=lambda w: w.x)

        # Точки разрыва: начало стены, концы окон, конец стены
        # Каждое окно создаёт 2 вертикальных разрыва + 2 горизонтальных (верх и низ окна)

        # Простой подход: строим сетку с пропусками
        # Разбиваем стену на колонки по X, в каждой колонке — сегменты по Y

        x_positions = [0.0]  # начало стены

        for win in sorted_wins:
            x1 = max(win.x, 0.0)
            x2 = min(win.x + win.width, wall_len)
            if x1 > x_positions[-1] + 0.01:
                x_positions.append(x1)
            if x2 > x_positions[-1] + 0.01 and x2 < wall_len - 0.01:
                x_positions.append(x2)

        if x_positions[-1] < wall_len - 0.01:
            x_positions.append(wall_len)

        # Для каждой колонки строим сегменты по высоте
        for col_idx in range(len(x_positions) - 1):
            x_start = x_positions[col_idx]
            x_end = x_positions[col_idx + 1]

            # Проверяем, попадает ли это окно в эту колонку
            col_windows = []
            for win in sorted_wins:
                win_x1 = win.x
                win_x2 = win.x + win.width
                # Пересечение с колонкой
                if not (win_x2 <= x_start or win_x1 >= x_end):
                    col_windows.append(win)

            # Точки по Y для этой колонки
            y_positions = [0.0]  # пол

            for win in col_windows:
                y1 = win.y
                y2 = win.y + win.height
                if y1 > y_positions[-1] + 0.01:
                    y_positions.append(y1)
                if y2 > y_positions[-1] + 0.01 and y2 < height - 0.01:
                    y_positions.append(y2)

            if y_positions[-1] < height - 0.01:
                y_positions.append(height)

            # Строим прямоугольники для каждого сегмента
            for row_idx in range(len(y_positions) - 1):
                y1 = y_positions[row_idx]
                y2 = y_positions[row_idx + 1]

                # Проверяем, является ли этот сегмент окном
                is_window = False
                for win in col_windows:
                    if (win.y - 0.01 <= y1 <= win.y + win.height + 0.01 and
                        win.x - 0.01 <= x_start <= win.x + win.width + 0.01):
                        is_window = True
                        break

                # Создаём вершины
                p0 = start + wall_dir * x_start + np.array([0, y1, 0])
                p1 = start + wall_dir * x_end + np.array([0, y1, 0])
                p2 = start + wall_dir * x_end + np.array([0, y2, 0])
                p3 = start + wall_dir * x_start + np.array([0, y2, 0])

                base = vertex_offset + len(vertices)
                vertices.extend([p0, p1, p2, p3])

                mat = "window" if is_window else "wall"
                # Два треугольника
                faces.extend([
                    (mat, base, base+1, base+2, wall_normal_idx),
                    (mat, base, base+2, base+3, wall_normal_idx)
                ])

        return vertices, faces, vertex_offset + len(vertices)

    def _build_floor(self, corners: List[np.ndarray], vertex_offset: int):
        """Построить пол (2 треугольника)."""
        v0, v1, v2, v3 = corners
        base = vertex_offset

        # Разбиваем четырёхугольник на 2 треугольника
        vertices = [v0, v1, v2, v3]
        faces = [
            (base, base+1, base+2, len(self.walls) + 1),   # v0, v1, v2
            (base, base+2, base+3, len(self.walls) + 1)    # v0, v2, v3
        ]

        return vertices, faces, vertex_offset + 4

    def _write_mtl(self, filename: str):
        """Записать файл материалов."""
        content = """# Room materials
newmtl wall
Ka 0.85 0.85 0.85
Kd 0.9 0.9 0.9
Ks 0.1 0.1 0.1
Ns 10

newmtl window
Ka 0.6 0.8 1.0
Kd 0.7 0.85 1.0
Ks 0.9 0.9 0.95
Ns 100
d 0.7

newmtl floor
Ka 0.5 0.4 0.3
Kd 0.6 0.5 0.4
Ks 0.1 0.1 0.1
Ns 5
"""
        with open(filename, 'w') as f:
            f.write(content)


def create_3d_model(room_dims: RoomDimensions, windows: List[Window], output_path: str):
    """Создать и сохранить 3D модель комнаты."""
    print("\n" + "="*50)
    print("ПОСТРОЕНИЕ 3D МОДЕЛИ")
    print("="*50)

    model = RoomModel3D(room_dims)
    model.build_walls(windows)
    model.export_obj(output_path)

    return model