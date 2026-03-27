import numpy as np
import cv2
from typing import List, Tuple, Dict
from dataclasses import dataclass
from pathlib import Path
from sklearn.cluster import KMeans
from room_detector import RoomDimensions, Window


@dataclass
class Wall3D:
    """3D стена с параметрами."""
    start: np.ndarray
    end: np.ndarray
    height: float
    normal: np.ndarray
    windows: List[Window] = None
    color: Tuple[float, float, float] = (0.9, 0.9, 0.9)

    def __post_init__(self):
        if self.windows is None:
            self.windows = []


class RoomModel3D:
    """Генератор 3D модели комнаты."""

    def __init__(self, room_dims: RoomDimensions):
        self.dims = room_dims
        self.walls = []
        self.floor = None
        self.floor_color = (0.5, 0.4, 0.3)

    def analyze_photo_colors(self, images: List[np.ndarray], windows_data: List[Dict]):
        """
        Улучшенный анализ цветов с фотографий.
        Учитывает освещение, убирает тени, находит настоящий цвет поверхности.
        """
        print("\n  Анализ цветов с фотографий...")

        wall_samples = {'left': [], 'right': [], 'top': [], 'bottom': []}
        floor_samples = []

        for i, img in enumerate(images):
            print(f"    Обработка фото {i + 1}...")
            h, w = img.shape[:2]

            # Конвертируем в LAB для лучшего восприятия цвета (независимо от яркости)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)

            # Убираем очень тёмные (тени) и очень светлые (блики) области
            mask = cv2.inRange(l, 30, 220)

            # Разбиваем на зоны
            zones = {
                'left': (slice(None), slice(0, w // 3)),
                'center': (slice(None), slice(w // 3, 2 * w // 3)),
                'right': (slice(None), slice(2 * w // 3, None)),
                'floor': (slice(2 * h // 3, None), slice(None))
            }

            for zone_name, (row_slice, col_slice) in zones.items():
                zone_lab = lab[row_slice, col_slice]
                zone_mask = mask[row_slice, col_slice]

                # Берём только не-теневые пиксели
                valid_pixels = zone_lab[zone_mask > 0]

                if len(valid_pixels) < 100:
                    continue

                # Находим доминирующий цвет
                dominant = self._get_robust_color(valid_pixels)

                if zone_name == 'floor':
                    floor_samples.append(dominant)
                else:
                    # Определяем стену
                    wall_map = {'left': 'left', 'center': 'top', 'right': 'right'}
                    wall_key = wall_map.get(zone_name)
                    if wall_key:
                        wall_samples[wall_key].append(dominant)

        # Усредняем с фильтрацией выбросов
        final_colors = {}

        for wall_name, colors in wall_samples.items():
            if colors:
                final_colors[wall_name] = self._merge_colors_robust(colors)
                r, g, b = final_colors[wall_name]
                print(f"    Стена {wall_name}: RGB({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})")
            else:
                final_colors[wall_name] = (0.9, 0.9, 0.9)

        # Пол — медиана с фильтрацией
        if floor_samples:
            self.floor_color = self._merge_colors_robust(floor_samples)
            r, g, b = self.floor_color
            print(f"    Пол: RGB({r * 255:.0f}, {g * 255:.0f}, {b * 255:.0f})")

        # Применяем к стенам
        wall_name_map = {'bottom': 'bottom', 'right': 'right', 'top': 'top', 'left': 'left'}
        for wall in self.walls:
            name = wall_name_map.get(self._get_wall_name(wall))
            if name and name in final_colors:
                wall.color = final_colors[name]

        return final_colors, self.floor_color

    def _get_robust_color(self, pixels: np.ndarray) -> Tuple[float, float, float]:
        """
        Находит устойчивый цвет с помощью K-means и выбора самого "среднего" кластера.
        """
        if len(pixels) < 50:
            return (0.9, 0.9, 0.9)

        # Уменьшаем выборку для скорости
        if len(pixels) > 5000:
            indices = np.random.choice(len(pixels), 5000, replace=False)
            sample = pixels[indices]
        else:
            sample = pixels

        # K-means на 3-5 кластеров
        try:
            kmeans = KMeans(n_clusters=min(5, len(sample) // 100), random_state=42, n_init=10)
            kmeans.fit(sample)

            # Находим самый большой кластер (основная поверхность)
            labels = kmeans.labels_
            counts = np.bincount(labels)
            dominant_label = np.argmax(counts)

            # Цвет в RGB
            lab_color = kmeans.cluster_centers_[dominant_label]
            # Конвертируем LAB -> RGB
            lab_pixel = np.uint8([[lab_color]])
            rgb = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0][0]

            # Нормализуем
            return tuple(rgb / 255.0)

        except Exception:
            # Fallback — медиана
            median = np.median(sample, axis=0)
            lab_pixel = np.uint8([[median]])
            rgb = cv2.cvtColor(lab_pixel, cv2.COLOR_LAB2RGB)[0][0]
            return tuple(rgb / 255.0)

    def _merge_colors_robust(self, colors: List[Tuple[float, float, float]]) -> Tuple[float, float, float]:
        """
        Усредняет несколько цветов с удалением выбросов.
        """
        if not colors:
            return (0.9, 0.9, 0.9)

        colors_array = np.array(colors)

        # Удаляем выбросы (цвета сильно отличающиеся от медианы)
        median = np.median(colors_array, axis=0)
        distances = np.linalg.norm(colors_array - median, axis=1)
        threshold = np.percentile(distances, 75)  # Убираем дальние 25%

        valid = colors_array[distances <= threshold]

        # Среднее оставшихся
        result = np.mean(valid, axis=0)
        return tuple(np.clip(result, 0, 1))

    def _get_wall_name(self, wall: Wall3D) -> str:
        """Определить имя стены по нормали."""
        normals = {
            'bottom': np.array([0, 0, -1]),
            'right': np.array([1, 0, 0]),
            'top': np.array([0, 0, 1]),
            'left': np.array([-1, 0, 0])
        }
        for name, n in normals.items():
            if np.allclose(wall.normal, n):
                return name
        return 'unknown'

    def build_walls(self, windows: List[Window]):
        """Построить 4 стены комнаты с вырезами под окна."""
        w, l, h = self.dims.width, self.dims.length, self.dims.height

        windows_by_wall = {'left': [], 'right': [], 'top': [], 'bottom': []}
        for win in windows:
            if win.wall in windows_by_wall:
                windows_by_wall[win.wall].append(win)

        corners = [
            np.array([0, 0, 0]),
            np.array([w, 0, 0]),
            np.array([w, 0, l]),
            np.array([0, 0, l])
        ]

        wall_configs = [
            ('bottom', corners[0], corners[1], np.array([0, 0, -1]), windows_by_wall['bottom']),
            ('right', corners[1], corners[2], np.array([1, 0, 0]), windows_by_wall['right']),
            ('top', corners[2], corners[3], np.array([0, 0, 1]), windows_by_wall['top']),
            ('left', corners[3], corners[0], np.array([-1, 0, 0]), windows_by_wall['left']),
        ]

        for name, start, end, normal, wall_windows in wall_configs:
            wall = Wall3D(start=start, end=end, height=h, normal=normal, windows=wall_windows)
            self.walls.append(wall)
            print(f"  Стена {name}: {np.linalg.norm(end - start):.2f}м, окон: {len(wall_windows)}")

        self.floor = {
            'corners': [corners[0], corners[1], corners[2], corners[3]],
            'normal': np.array([0, 1, 0])
        }

    def export_obj(self, filename: str, images: List[np.ndarray] = None, windows_data: List[Dict] = None):
        """Экспорт в Wavefront OBJ с цветами из фото."""

        if images and windows_data:
            self.analyze_photo_colors(images, windows_data)

        all_vertices = []
        all_faces = []
        vertex_offset = 1

        mtl_filename = filename.replace('.obj', '.mtl')

        lines = [
            "# Room 3D Model - Photo colors",
            f"mtllib {Path(mtl_filename).name}",
            ""
        ]

        for i, wall in enumerate(self.walls):
            lines.append(f"# Wall {i + 1}")
            verts, faces, vertex_offset = self._build_wall_with_holes(wall, vertex_offset, i)
            all_vertices.extend(verts)
            all_faces.extend(faces)

        lines.append("# Floor")
        verts, faces, vertex_offset = self._build_floor(self.floor['corners'], vertex_offset)
        all_vertices.extend(verts)
        all_faces.extend([("floor", *f) for f in faces])

        for v in all_vertices:
            lines.append(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}")

        lines.extend(["", "vt 0.0 0.0", "vt 1.0 0.0", "vt 1.0 1.0", "vt 0.0 1.0"])

        lines.append("")
        for wall in self.walls:
            lines.append(f"vn {wall.normal[0]:.4f} {wall.normal[1]:.4f} {wall.normal[2]:.4f}")
        lines.append("vn 0.0 1.0 0.0")

        lines.append("")
        current_mtl = None

        for face in all_faces:
            mat = face[0]
            if mat != current_mtl:
                current_mtl = mat
                lines.append(f"usemtl {mat}")

            if mat == "floor":
                ni = len(self.walls) + 1
            else:
                ni = 1

            if len(face) == 5:
                _, v1, v2, v3, ni_face = face
                lines.append(f"f {v1}/{1}/{ni_face} {v2}/{2}/{ni_face} {v3}/{3}/{ni_face}")
            elif len(face) == 6:
                _, v1, v2, v3, v4, ni_face = face
                lines.append(f"f {v1}/{1}/{ni_face} {v2}/{2}/{ni_face} {v3}/{3}/{ni_face} {v4}/{4}/{ni_face}")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))

        self._write_mtl_colored(mtl_filename)

        print(f"\n✓ 3D модель сохранена: {filename}")
        print(f"  Вершин: {len(all_vertices)}")
        print(f"  Граней: {len(all_faces)}")

    def _build_wall_with_holes(self, wall: Wall3D, vertex_offset: int, wall_idx: int):
        """Построить стену с вырезами под окна."""
        vertices = []
        faces = []

        start, end = wall.start, wall.end
        height = wall.height
        wall_vec = end - start
        wall_len = np.linalg.norm(wall_vec)
        wall_dir = wall_vec / wall_len if wall_len > 0 else np.array([1, 0, 0])

        wall_normal_idx = wall_idx + 1

        if not wall.windows:
            v0, v1, v2, v3 = start, end, end + np.array([0, height, 0]), start + np.array([0, height, 0])
            base = vertex_offset
            vertices = [v0, v1, v2, v3]
            faces = [
                (f"wall_{wall_idx}", base, base + 1, base + 2, wall_normal_idx),
                (f"wall_{wall_idx}", base, base + 2, base + 3, wall_normal_idx)
            ]
            return vertices, faces, vertex_offset + 4

        sorted_wins = sorted(wall.windows, key=lambda w: w.x)
        x_positions = [0.0]

        for win in sorted_wins:
            x1, x2 = max(win.x, 0.0), min(win.x + win.width, wall_len)
            if x1 > x_positions[-1] + 0.01:
                x_positions.append(x1)
            if x2 > x_positions[-1] + 0.01 and x2 < wall_len - 0.01:
                x_positions.append(x2)

        if x_positions[-1] < wall_len - 0.01:
            x_positions.append(wall_len)

        for col_idx in range(len(x_positions) - 1):
            x_start, x_end = x_positions[col_idx], x_positions[col_idx + 1]

            col_windows = [w for w in sorted_wins
                           if not (w.x + w.width <= x_start or w.x >= x_end)]

            y_positions = [0.0]
            for win in col_windows:
                y1, y2 = win.y, win.y + win.height
                if y1 > y_positions[-1] + 0.01:
                    y_positions.append(y1)
                if y2 > y_positions[-1] + 0.01 and y2 < height - 0.01:
                    y_positions.append(y2)

            if y_positions[-1] < height - 0.01:
                y_positions.append(height)

            for row_idx in range(len(y_positions) - 1):
                y1, y2 = y_positions[row_idx], y_positions[row_idx + 1]

                is_window = any(
                    w.y - 0.01 <= y1 <= w.y + w.height + 0.01 and
                    w.x - 0.01 <= x_start <= w.x + w.width + 0.01
                    for w in col_windows
                )

                p0 = start + wall_dir * x_start + np.array([0, y1, 0])
                p1 = start + wall_dir * x_end + np.array([0, y1, 0])
                p2 = start + wall_dir * x_end + np.array([0, y2, 0])
                p3 = start + wall_dir * x_start + np.array([0, y2, 0])

                base = vertex_offset + len(vertices)
                vertices.extend([p0, p1, p2, p3])

                mat = f"window_{wall_idx}" if is_window else f"wall_{wall_idx}"
                faces.extend([
                    (mat, base, base + 1, base + 2, wall_normal_idx),
                    (mat, base, base + 2, base + 3, wall_normal_idx)
                ])

        return vertices, faces, vertex_offset + len(vertices)

    def _build_floor(self, corners: List[np.ndarray], vertex_offset: int):
        """Построить пол."""
        v0, v1, v2, v3 = corners
        base = vertex_offset

        vertices = [v0, v1, v2, v3]
        faces = [
            (base, base + 1, base + 2, len(self.walls) + 1),
            (base, base + 2, base + 3, len(self.walls) + 1)
        ]

        return vertices, faces, vertex_offset + 4

    def _write_mtl_colored(self, filename: str):
        """Записать MTL с реальными цветами."""
        lines = ["# Room materials - Photo colors"]

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
            wr = min(r + 0.3, 1.0)
            wg = min(g + 0.3, 1.0)
            wb = min(b + 0.4, 1.0)
            lines.append(f"""
newmtl window_{i}
Ka {wr:.3f} {wg:.3f} {wb:.3f}
Kd {wr:.3f} {wg:.3f} {wb:.3f}
Ks 0.9 0.9 0.95
Ns 100
d 0.5
illum 4""")

        # Пол
        fr, fg, fb = self.floor_color
        lines.append(f"""
newmtl floor
Ka {fr:.3f} {fg:.3f} {fb:.3f}
Kd {fr:.3f} {fg:.3f} {fb:.3f}
Ks 0.1 0.1 0.1
Ns 5
illum 2
""")

        with open(filename, 'w') as f:
            f.write('\n'.join(lines))


def create_3d_model(room_dims: RoomDimensions, windows: List[Window], output_path: str,
                    images: List[np.ndarray] = None, windows_data: List[Dict] = None):
    """Создать 3D модель с цветами из фото."""
    print("ПОСТРОЕНИЕ 3D МОДЕЛИ")

    model = RoomModel3D(room_dims)
    model.build_walls(windows)
    model.export_obj(output_path, images=images, windows_data=windows_data)

    return model