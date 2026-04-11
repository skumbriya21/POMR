import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class FurnitureType(Enum):
    """Типы мебели и предметов."""
    TABLE = "table"  # Стол
    CHAIR = "chair"  # Стул
    SOFA = "sofa"  # Диван/кресло
    BED = "bed"  # Кровать
    CABINET = "cabinet"  # Шкаф/комод
    UNKNOWN = "unknown"  # Неопознанный предмет


@dataclass
class DetectedFurniture:
    """Обнаруженный предмет мебели."""
    x: int  # X на изображении
    y: int  # Y на изображении
    width: int  # Ширина в пикселях
    height: int  # Высота в пикселях
    depth_estimate: float  # Оценочная глубина (относительная)
    furniture_type: FurnitureType
    confidence: float
    is_valid: bool = True

    # 3D параметры (в метрах, будут рассчитаны позже)
    width_3d: float = 0.0
    height_3d: float = 0.0
    depth_3d: float = 0.0
    position_3d: np.ndarray = None  # [x, y, z] в координатах комнаты


@dataclass
class BoundingBox3D:
    """3D ограничивающий параллелепипед (коробка)."""
    center: np.ndarray  # [x, y, z]
    dimensions: np.ndarray  # [width, height, depth]
    rotation: float = 0.0  # Поворот вокруг Y (вертикальной оси) в радианах

    @property
    def min_corner(self) -> np.ndarray:
        return self.center - self.dimensions / 2

    @property
    def max_corner(self) -> np.ndarray:
        return self.center + self.dimensions / 2


class FurnitureDetectorCV:
    """
    Детектор мебели на фотографиях с использованием компьютерного зрения.
    Использует комбинацию методов: детекция контуров, анализ формы,
    сегментация по цвету и глубине (если доступна).
    """

    # Стандартные размеры мебели в метрах (для эвристик)
    TYPICAL_SIZES = {
        FurnitureType.TABLE: {'width': (0.8, 1.6), 'height': (0.7, 0.8), 'depth': (0.6, 1.2)},
        FurnitureType.CHAIR: {'width': (0.4, 0.6), 'height': (0.8, 1.0), 'depth': (0.4, 0.6)},
        FurnitureType.SOFA: {'width': (1.5, 2.5), 'height': (0.7, 0.9), 'depth': (0.8, 1.0)},
        FurnitureType.BED: {'width': (1.4, 2.0), 'height': (0.4, 0.6), 'depth': (1.9, 2.2)},
        FurnitureType.CABINET: {'width': (0.6, 1.2), 'height': (1.2, 2.0), 'depth': (0.4, 0.6)},
        FurnitureType.UNKNOWN: {'width': (0.3, 2.0), 'height': (0.3, 2.0), 'depth': (0.3, 2.0)},
    }

    def __init__(self, min_area=5000, max_area=500000):
        self.min_area = min_area
        self.max_area = max_area

        # Параметры для детекции разных типов мебели
        self.furniture_params = {
            FurnitureType.TABLE: {
                'aspect_ratio': (1.0, 2.5),  # Ширина/высота на фото
                'fill_ratio': (0.3, 0.9),  # Заполнение контура
                'bottom_position': (0.5, 1.0),  # Положение относительно низа кадра
                'color_variance': (10, 100),  # Вариативность цвета (текстура дерева/пластика)
            },
            FurnitureType.CHAIR: {
                'aspect_ratio': (0.5, 1.2),
                'fill_ratio': (0.2, 0.8),
                'bottom_position': (0.6, 1.0),
                'color_variance': (5, 80),
            },
            FurnitureType.SOFA: {
                'aspect_ratio': (1.5, 4.0),
                'fill_ratio': (0.4, 0.95),
                'bottom_position': (0.5, 1.0),
                'color_variance': (15, 120),
            },
            FurnitureType.BED: {
                'aspect_ratio': (1.0, 3.0),
                'fill_ratio': (0.5, 0.98),
                'bottom_position': (0.4, 0.9),
                'color_variance': (5, 60),
            },
            FurnitureType.CABINET: {
                'aspect_ratio': (0.3, 1.0),
                'fill_ratio': (0.6, 0.95),
                'bottom_position': (0.3, 1.0),
                'color_variance': (10, 80),
            },
        }

        # Фоновое вычитание для статичных камер (опционально)
        self.background_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=False
        )

    def detect_furniture(self, image: np.ndarray,
                         room_floor_y: Optional[float] = None) -> List[DetectedFurniture]:
        """
        Детекция мебели на одном изображении.

        Args:
            image: BGR изображение
            room_floor_y: Оценочная Y-координата пола на изображении (0-1)

        Returns:
            Список обнаруженной мебели
        """
        img_height, img_width = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 1. Предварительная обработка
        # Улучшение контраста
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray_eq = clahe.apply(gray)

        # 2. Детекция границ и контуров
        edges = cv2.Canny(gray_eq, 30, 150)

        # Морфологические операции для соединения разрозненных линий
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)

        # 3. Поиск контуров
        contours, _ = cv2.findContours(
            edges_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        furniture_list = []

        # 4. Анализ каждого контура
        for contour in contours:
            area = cv2.contourArea(contour)
            if not (self.min_area < area < self.max_area):
                continue

            # Получаем ограничивающий прямоугольник
            x, y, w, h = cv2.boundingRect(contour)

            # Аппроксимация контура для проверки "прямоугольности"
            epsilon = 0.05 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Проверяем соотношение сторон
            aspect_ratio = w / float(h) if h > 0 else 0

            # Оценка позиции относительно пола
            bottom_relative = (y + h) / img_height
            if room_floor_y and abs(bottom_relative - room_floor_y) > 0.15:
                # Слишком далеко от пола — возможно не мебель
                pass  # Не фильтруем строго, просто отмечаем

            # Анализ ROI
            roi = image[y:y + h, x:x + w]
            roi_gray = gray[y:y + h, x:x + w]

            # Оценка текстуры и цвета
            color_variance = np.std(roi_gray)

            # Оценка заполненности контура
            rect_area = w * h
            fill_ratio = area / rect_area if rect_area > 0 else 0

            # Проверка на тени (мебель обычно отбрасывает тень)
            shadow_detected = self._detect_shadow(gray, x, y, w, h, img_height)

            # Определяем тип мебели
            furniture_type, confidence = self._classify_furniture(
                aspect_ratio, fill_ratio, bottom_relative,
                color_variance, shadow_detected, w, h, img_width, img_height
            )

            # Оценка глубины (простая эвристика: ниже на кадре = ближе)
            depth_estimate = self._estimate_depth(y, h, img_height, bottom_relative)

            if confidence > 0.3:
                furniture = DetectedFurniture(
                    x=x, y=y, width=w, height=h,
                    depth_estimate=depth_estimate,
                    furniture_type=furniture_type,
                    confidence=confidence,
                    is_valid=confidence > 0.5
                )
                furniture_list.append(furniture)

        # 5. Удаление дубликатов и пересечений
        furniture_list = self._remove_overlapping(furniture_list)

        # 6. Сортировка по глубине (от ближних к дальним)
        furniture_list.sort(key=lambda f: f.depth_estimate, reverse=True)

        return furniture_list

    def _detect_shadow(self, gray: np.ndarray, x: int, y: int,
                       w: int, h: int, img_height: int) -> bool:
        """Детекция тени под объектом (признак мебели)."""
        # Проверяем область под объектом
        shadow_y_start = min(y + h, img_height - 1)
        shadow_y_end = min(y + h + int(h * 0.3), img_height)

        if shadow_y_end <= shadow_y_start:
            return False

        shadow_region = gray[shadow_y_start:shadow_y_end, x:x + w]
        object_region = gray[y:y + h, x:x + w]

        if shadow_region.size == 0 or object_region.size == 0:
            return False

        shadow_brightness = np.mean(shadow_region)
        object_brightness = np.mean(object_region)

        # Тень темнее объекта
        return shadow_brightness < object_brightness * 0.7

    def _classify_furniture(self, aspect_ratio: float, fill_ratio: float,
                            bottom_pos: float, color_variance: float,
                            shadow_detected: bool, w: int, h: int,
                            img_w: int, img_h: int) -> Tuple[FurnitureType, float]:
        """Классификация типа мебели на основе признаков."""

        scores = {}

        for ftype, params in self.furniture_params.items():
            score = 0.0

            # Соотношение сторон
            ar_min, ar_max = params['aspect_ratio']
            if ar_min <= aspect_ratio <= ar_max:
                ar_score = 1.0 - abs(aspect_ratio - (ar_min + ar_max) / 2) / ((ar_max - ar_min) / 2)
                score += ar_score * 0.25
            else:
                score -= 0.2

            # Заполнение
            fr_min, fr_max = params['fill_ratio']
            if fr_min <= fill_ratio <= fr_max:
                score += 0.2

            # Позиция относительно пола
            bp_min, bp_max = params['bottom_position']
            if bp_min <= bottom_pos <= bp_max:
                score += 0.2

            # Вариативность цвета (текстура)
            cv_min, cv_max = params['color_variance']
            if cv_min <= color_variance <= cv_max:
                score += 0.15

            # Наличие тени
            if shadow_detected:
                score += 0.1

            # Размер относительно изображения
            relative_size = (w * h) / (img_w * img_h)
            if 0.02 < relative_size < 0.4:
                score += 0.1

            scores[ftype] = max(0, score)

        # Выбираем лучший тип
        best_type = max(scores, key=scores.get)
        best_score = scores[best_type]

        # Если уверенность низкая — unknown
        if best_score < 0.3:
            return FurnitureType.UNKNOWN, best_score

        return best_type, min(best_score, 1.0)

    def _estimate_depth(self, y: int, h: int, img_height: int,
                        bottom_pos: float) -> float:
        """
        Оценка относительной глубины объекта.
        Простая перспективная модель: объекты ниже на кадре ближе.
        """
        # Нормализуем позицию (0 = далеко/верх, 1 = близко/низ)
        depth = bottom_pos

        # Корректировка на размер (большие объекты обычно ближе)
        relative_height = h / img_height
        depth = depth * 0.7 + relative_height * 0.3

        return min(max(depth, 0.0), 1.0)

    def _remove_overlapping(self, furniture: List[DetectedFurniture],
                            iou_threshold: float = 0.3) -> List[DetectedFurniture]:
        """Удаление пересекающихся детекций (NMS)."""
        if len(furniture) <= 1:
            return furniture

        # Сортируем по уверенности
        sorted_furniture = sorted(furniture, key=lambda f: f.confidence, reverse=True)

        keep = []
        for f1 in sorted_furniture:
            overlap = False
            for f2 in keep:
                if self._iou(f1, f2) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(f1)

        return keep

    def _iou(self, f1: DetectedFurniture, f2: DetectedFurniture) -> float:
        """Вычисление IoU двух прямоугольников."""
        x1 = max(f1.x, f2.x)
        y1 = max(f1.y, f2.y)
        x2 = min(f1.x + f1.width, f2.x + f2.width)
        y2 = min(f1.y + f1.height, f2.y + f2.height)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = f1.width * f1.height
        area2 = f2.width * f2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def analyze_multiple_images(self, images: List[np.ndarray],
                                camera_poses: Optional[List[np.ndarray]] = None
                                ) -> List[Dict]:
        """
        Анализ нескольких изображений для 3D реконструкции мебели.

        Args:
            images: Список изображений
            camera_poses: Позы камер (опционально, для триангуляции)

        Returns:
            Список агрегированной информации о мебели
        """
        all_detections = []

        for i, img in enumerate(images):
            print(f"  Фото {i + 1}: анализ мебели...")
            detections = self.detect_furniture(img)
            print(f"    Найдено {len(detections)} объектов")

            for det in detections:
                all_detections.append({
                    'photo_idx': i,
                    'furniture': det,
                    'camera_pose': camera_poses[i] if camera_poses else None
                })

        # Группировка по типу и примерной позиции
        grouped = self._group_detections(all_detections)

        return grouped

    def _group_detections(self, detections: List[Dict]) -> List[Dict]:
        """Группировка детекций одного объекта с разных фото."""
        if not detections:
            return []

        # Простая кластеризация по типу и относительной позиции
        groups = []

        for det in detections:
            furniture = det['furniture']
            matched = False

            for group in groups:
                # Проверяем совпадение типа
                if group['type'] != furniture.furniture_type:
                    continue

                # Проверяем близость позиций (если есть camera_pose)
                if det['camera_pose'] is not None and group.get('camera_poses'):
                    # TODO: Реализовать триангуляцию для точной позиции
                    pass

                # Простая проверка по photo_idx (разные фото)
                if det['photo_idx'] != group['photo_idx']:
                    group['detections'].append(det)
                    matched = True
                    break

            if not matched:
                groups.append({
                    'type': furniture.furniture_type,
                    'photo_idx': det['photo_idx'],
                    'detections': [det],
                    'camera_poses': [det['camera_pose']] if det['camera_pose'] else []
                })

        # Формируем финальный результат
        result = []
        for group in groups:
            if len(group['detections']) >= 1:
                # Берем лучшую детекцию
                best = max(group['detections'],
                           key=lambda d: d['furniture'].confidence)

                result.append({
                    'type': group['type'],
                    'confidence': best['furniture'].confidence,
                    'dimensions_2d': (best['furniture'].width, best['furniture'].height),
                    'depth_estimate': best['furniture'].depth_estimate,
                    'verified': len(group['detections']) >= 2
                })

        return result


class Furniture3DReconstructor:
    """
    Реконструкция 3D коробок (bounding boxes) для мебели.
    """

    def __init__(self, room_dims):
        self.room_width = room_dims.width
        self.room_length = room_dims.length
        self.room_height = room_dims.height

        # Сетка для размещения объектов (избегаем пересечений)
        self.occupied_regions = []

    def reconstruct_furniture_3d(self, detected_furniture: List[Dict],
                                 windows: List, doors: List) -> List[BoundingBox3D]:
        """
        Создание 3D коробок для обнаруженной мебели.

        Args:
            detected_furniture: Результат FurnitureDetectorCV.analyze_multiple_images()
            windows: Окна комнаты (для избегания пересечений)
            doors: Двери комнаты (для избегания пересечений)

        Returns:
            Список 3D коробок
        """
        boxes_3d = []

        print(f"\nРеконструкция 3D мебели: {len(detected_furniture)} объектов")

        for i, furn in enumerate(detected_furniture, 1):
            ftype = furn['type']

            # Определяем размеры на основе типа
            dims = self._estimate_dimensions(ftype, furn.get('dimensions_2d'))

            # Находим свободную позицию в комнате
            position = self._find_placement_position(
                dims, windows, doors, boxes_3d
            )

            if position is None:
                print(f"  ! Не удалось разместить {ftype.value} #{i}")
                continue

            # Создаем коробку
            box = BoundingBox3D(
                center=position,
                dimensions=np.array([dims['width'], dims['height'], dims['depth']]),
                rotation=0.0  # Можно добавить ориентацию позже
            )

            boxes_3d.append(box)
            self.occupied_regions.append(self._get_region_2d(box))

            print(f"  ✓ {ftype.value} #{i}: {dims['width']:.2f}×{dims['height']:.2f}×{dims['depth']:.2f}м "
                  f"на позиции ({position[0]:.2f}, {position[1]:.2f}, {position[2]:.2f})")

        return boxes_3d

    def _estimate_dimensions(self, ftype: FurnitureType,
                             dims_2d: Optional[Tuple[int, int]]) -> Dict:
        """Оценка 3D размеров на основе типа и 2D проекции."""
        sizes = FurnitureDetectorCV.TYPICAL_SIZES.get(ftype,
                                                      FurnitureDetectorCV.TYPICAL_SIZES[FurnitureType.UNKNOWN])

        # Берем средние значения из диапазонов
        width = (sizes['width'][0] + sizes['width'][1]) / 2
        height = (sizes['height'][0] + sizes['height'][1]) / 2
        depth = (sizes['depth'][0] + sizes['depth'][1]) / 2

        # Корректировка по 2D размеру (если есть)
        if dims_2d:
            w_2d, h_2d = dims_2d
            aspect_2d = w_2d / h_2d if h_2d > 0 else 1.0

            # Если объект шире чем выше, возможно это вид сбоку
            if aspect_2d > 2.0:
                # Меняем width и depth местами (поворот)
                width, depth = depth, width

        return {
            'width': width,
            'height': height,
            'depth': depth
        }

    def _find_placement_position(self, dims: Dict, windows, doors,
                                 existing_boxes: List[BoundingBox3D]) -> Optional[np.ndarray]:
        """
        Поиск свободной позиции для размещения мебели.
        Использует эвристики для типичного расположения.
        """
        width, depth = dims['width'], dims['depth']
        height = dims['height']

        # Попытки размещения (приоритетные зоны)
        candidates = self._generate_candidate_positions(width, depth)

        for pos_x, pos_z in candidates:
            # Проверяем высоту (Y) — мебель стоит на полу
            pos_y = height / 2  # Центр коробки на половине высоты

            position = np.array([pos_x, pos_y, pos_z])

            # Проверяем коллизии
            if self._check_collision(position, np.array([width, height, depth]),
                                     windows, doors, existing_boxes):
                continue

            # Проверяем границы комнаты
            if not self._is_inside_room(position, width, depth):
                continue

            return position

        return None

    def _generate_candidate_positions(self, obj_width: float,
                                      obj_depth: float) -> List[Tuple[float, float]]:
        """Генерация кандидатных позиций с приоритетами."""
        candidates = []

        margin = 0.5  # Отступ от стен

        # Сетка позиций
        grid_x = np.linspace(margin + obj_width / 2,
                             self.room_width - margin - obj_width / 2, 5)
        grid_z = np.linspace(margin + obj_depth / 2,
                             self.room_length - margin - obj_depth / 2, 5)

        # Центр комнаты (низкий приоритет для большинства мебели)
        center_x, center_z = self.room_width / 2, self.room_length / 2

        # Углы комнаты (высокий приоритет для шкафов)
        corners = [
            (margin + obj_width / 2, margin + obj_depth / 2),
            (self.room_width - margin - obj_width / 2, margin + obj_depth / 2),
            (margin + obj_width / 2, self.room_length - margin - obj_depth / 2),
            (self.room_width - margin - obj_width / 2, self.room_length - margin - obj_depth / 2),
        ]

        # У стен (средний приоритет)
        wall_positions = []
        for x in grid_x:
            wall_positions.append((x, margin + obj_depth / 2))
            wall_positions.append((x, self.room_length - margin - obj_depth / 2))
        for z in grid_z:
            wall_positions.append((margin + obj_width / 2, z))
            wall_positions.append((self.room_width - margin - obj_width / 2, z))

        # Комбинируем с приоритетами
        candidates.extend(corners)
        candidates.extend(wall_positions)

        # Добавляем случайные позиции в центре
        np.random.seed(42)
        for _ in range(10):
            rx = np.random.uniform(margin, self.room_width - margin)
            rz = np.random.uniform(margin, self.room_length - margin)
            candidates.append((rx, rz))

        return candidates

    def _check_collision(self, position: np.ndarray, dimensions: np.ndarray,
                         windows, doors, existing_boxes: List[BoundingBox3D]) -> bool:
        """Проверка коллизий с окнами, дверями и другой мебелью."""
        new_min = position - dimensions / 2
        new_max = position + dimensions / 2

        # Проверка с окнами (не размещаем мебель вплотную)
        window_margin = 0.3
        for win in windows:
            if win.wall in ['left', 'right']:
                win_x = 0 if win.wall == 'left' else self.room_width
                win_z_start = win.x
                win_z_end = win.x + win.width
                win_y_bottom = win.y
                win_y_top = win.y + win.height

                # Проверяем пересечение по Z и Y
                if (new_min[2] < win_z_end + window_margin and
                        new_max[2] > win_z_start - window_margin and
                        new_max[1] > win_y_bottom and
                        new_min[1] < win_y_top):
                    # Проверяем близость к стене с окном
                    if win.wall == 'left' and new_min[0] < window_margin:
                        return True
                    if win.wall == 'right' and new_max[0] > self.room_width - window_margin:
                        return True
            else:
                win_z = 0 if win.wall == 'bottom' else self.room_length
                win_x_start = win.x
                win_x_end = win.x + win.width

                if (new_min[0] < win_x_end + window_margin and
                        new_max[0] > win_x_start - window_margin and
                        new_max[1] > win.y and
                        new_min[1] < win.y + win.height):
                    if win.wall == 'bottom' and new_min[2] < window_margin:
                        return True
                    if win.wall == 'top' and new_max[2] > self.room_length - window_margin:
                        return True

        # Проверка с дверями
        door_margin = 0.5
        for door in doors:
            if door.wall in ['left', 'right']:
                door_z_start = door.x
                door_z_end = door.x + door.width
                if (new_min[2] < door_z_end + door_margin and
                        new_max[2] > door_z_start - door_margin):
                    if door.wall == 'left' and new_min[0] < door_margin:
                        return True
                    if door.wall == 'right' and new_max[0] > self.room_width - door_margin:
                        return True
            else:
                door_x_start = door.x
                door_x_end = door.x + door.width
                if (new_min[0] < door_x_end + door_margin and
                        new_max[0] > door_x_start - door_margin):
                    if door.wall == 'bottom' and new_min[2] < door_margin:
                        return True
                    if door.wall == 'top' and new_max[2] > self.room_length - door_margin:
                        return True

        # Проверка с другой мебелью
        furniture_margin = 0.2
        for box in existing_boxes:
            existing_min = box.min_corner - furniture_margin
            existing_max = box.max_corner + furniture_margin

            if (new_min[0] < existing_max[0] and new_max[0] > existing_min[0] and
                    new_min[1] < existing_max[1] and new_max[1] > existing_min[1] and
                    new_min[2] < existing_max[2] and new_max[2] > existing_min[2]):
                return True

        return False

    def _is_inside_room(self, position: np.ndarray, width: float, depth: float) -> bool:
        """Проверка, что объект полностью внутри комнаты."""
        margin = 0.05
        return (
                position[0] - width / 2 >= margin and
                position[0] + width / 2 <= self.room_width - margin and
                position[2] - depth / 2 >= margin and
                position[2] + depth / 2 <= self.room_length - margin and
                position[1] >= 0 and position[1] <= self.room_height
        )

    def _get_region_2d(self, box: BoundingBox3D) -> Tuple[float, float, float, float]:
        """Получение 2D проекции региона (для отслеживания занятых зон)."""
        return (box.center[0] - box.dimensions[0] / 2,
                box.center[2] - box.dimensions[2] / 2,
                box.center[0] + box.dimensions[0] / 2,
                box.center[2] + box.dimensions[2] / 2)


def map_furniture_to_3d(detected_furniture: List[Dict],
                        room_dims,
                        windows, doors) -> List[BoundingBox3D]:
    """
    Удобная функция для конвертации детекций в 3D коробки.
    """
    reconstructor = Furniture3DReconstructor(room_dims)
    return reconstructor.reconstruct_furniture_3d(detected_furniture, windows, doors)