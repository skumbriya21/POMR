import cv2
import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass


@dataclass
class DetectedDoor:
    """Обнаруженная дверь на фото."""
    x: int
    y: int
    width: int
    height: int
    confidence: float
    wall_position: str  # 'left', 'right', 'center' (для определения стены)
    is_valid: bool = True
    has_glass: bool = False
    is_open: bool = False


class DoorDetectorCV:
    """
    Детектор дверей на фотографиях с использованием компьютерного зрения.
    """

    def __init__(self, min_area=30000, max_area=300000):
        """
        Args:
            min_area: Минимальная площадь двери в пикселях
            max_area: Максимальная площадь двери в пикселях
        """
        self.min_area = min_area
        self.max_area = max_area

        # Типичные параметры двери
        self.typical_door = {
            'aspect_ratio': (1.5, 3.5),  # Высота/ширина
            'relative_height': (0.5, 0.9),  # Относительная высота на фото
            'wall_margin': 0.1  # Отступ от края стены
        }

    def detect_doors(self, image: np.ndarray) -> List[DetectedDoor]:
        """
        Обнаружение дверей на изображении.

        Args:
            image: Изображение в формате BGR (OpenCV)

        Returns:
            Список обнаруженных дверей
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        img_height, img_width = image.shape[:2]

        # 1. Детекция вертикальных и горизонтальных линий (каркас двери)
        edges = cv2.Canny(gray, 50, 150)

        # 2. Поиск прямоугольных контуров
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        doors = []

        # 3. Анализ яркости для поиска проемов (дверь обычно темнее стены или светлее)
        brightness = np.mean(gray)

        # 4. Поиск областей с текстурой дерева/металла (для дверей)
        # Используем градиенты для детекции вертикальных линий
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(sobel_x ** 2 + sobel_y ** 2)

        # 5. Анализ каждого контура
        for contour in contours:
            # Аппроксимация контура
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Дверь должна иметь 4 угла (прямоугольник)
            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            area = w * h

            # Проверка площади
            if not (self.min_area < area < self.max_area):
                continue

            # Проверка соотношения сторон (дверь выше чем шире)
            aspect = h / float(w) if w > 0 else 0
            if not (self.typical_door['aspect_ratio'][0] < aspect < self.typical_door['aspect_ratio'][1]):
                continue

            # Проверка относительной высоты (дверь от пола до потолка или чуть ниже)
            relative_y_top = y / img_height
            relative_y_bottom = (y + h) / img_height

            if relative_y_bottom < 0.6:  # Дверь не может быть слишком высоко
                continue

            # Проверка заполненности контура
            rect_area = w * h
            contour_area = cv2.contourArea(contour)
            fill_ratio = contour_area / rect_area if rect_area > 0 else 0

            # Дверь должна быть достаточно заполнена (не просто рамка)
            if fill_ratio < 0.4:
                continue

            # Определение стены по горизонтальной позиции
            center_x = x + w / 2
            if center_x < img_width * 0.25:
                wall_position = 'left'
            elif center_x > img_width * 0.75:
                wall_position = 'right'
            else:
                wall_position = 'center'

            # Анализ ROI для определения типа двери
            roi = gray[y:y + h, x:x + w]
            roi_magnitude = magnitude[y:y + h, x:x + w]

            # 6. Детекция стеклянной двери (более светлая, меньше градиентов)
            mean_brightness = np.mean(roi)
            mean_gradient = np.mean(roi_magnitude)

            has_glass = False
            if mean_brightness > brightness * 1.1:  # Светлее среднего
                has_glass = True

            # 7. Детекция открытой двери (поиск угла)
            is_open = self._detect_open_door(contour, approx, w, h)

            # 8. Расчет уверенности
            confidence = self._calculate_confidence(
                aspect, fill_ratio, mean_gradient,
                relative_y_top, relative_y_bottom, area
            )

            is_valid = confidence > 0.35

            doors.append(DetectedDoor(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                wall_position=wall_position,
                is_valid=is_valid,
                has_glass=has_glass,
                is_open=is_open
            ))

        # Сортировка по уверенности и удаление пересекающихся
        doors.sort(key=lambda d: d.confidence, reverse=True)
        valid_doors = [d for d in doors if d.is_valid]

        # Если не нашли валидных, берем лучший кандидат
        if len(valid_doors) == 0 and len(doors) > 0:
            best = doors[0]
            best.confidence *= 0.6
            best.is_valid = True
            valid_doors = [best]

        return self._remove_overlapping(valid_doors)

    def _detect_open_door(self, contour, approx, w, h) -> bool:
        """Определение, открыта ли дверь (по форме контура)."""
        # Для открытой двери контур не будет идеальным прямоугольником
        # или будет иметь дополнительный выступ

        contour_area = cv2.contourArea(contour)
        rect_area = w * h

        if rect_area == 0:
            return False

        fill_ratio = contour_area / rect_area

        # Если заполнение низкое - возможно дверь открыта
        if fill_ratio < 0.5 and fill_ratio > 0.2:
            return True

        return False

    def _calculate_confidence(self, aspect, fill_ratio, gradient,
                              y_top, y_bottom, area) -> float:
        """Расчет уверенности, что найденный объект - дверь."""

        # Оценка по соотношению сторон
        ideal_aspect = 2.5  # Идеальное для двери
        aspect_score = 1.0 - min(abs(aspect - ideal_aspect) / 2.0, 1.0)

        # Оценка по заполнению
        fill_score = 1.0 - abs(fill_ratio - 0.85) / 0.3
        fill_score = max(0, min(1, fill_score))

        # Оценка по позиции на фото (дверь должна быть от пола)
        position_score = 1.0
        if y_bottom < 0.7:
            position_score = y_bottom / 0.7
        if y_top > 0.1:
            position_score *= 0.8

        # Оценка по текстуре (дверь имеет текстуру, но не слишком много градиентов)
        gradient_score = min(gradient / 50, 1.0)

        # Оценка по размеру
        size_score = min(area / 150000, 1.0)

        # Итоговая оценка
        confidence = (
                aspect_score * 0.25 +
                fill_score * 0.25 +
                position_score * 0.2 +
                gradient_score * 0.15 +
                size_score * 0.15
        )

        return min(confidence, 1.0)

    def _remove_overlapping(self, doors: List[DetectedDoor], iou_threshold=0.4) -> List[DetectedDoor]:
        """Удаление пересекающихся дверей."""
        if len(doors) <= 1:
            return doors

        keep = []
        for d1 in doors:
            overlap = False
            for d2 in keep:
                if self._iou(d1, d2) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(d1)
        return keep

    def _iou(self, d1: DetectedDoor, d2: DetectedDoor) -> float:
        """Вычисление IoU (Intersection over Union) для двух дверей."""
        x1 = max(d1.x, d2.x)
        y1 = max(d1.y, d2.y)
        x2 = min(d1.x + d1.width, d2.x + d2.width)
        y2 = min(d1.y + d1.height, d2.y + d2.height)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = d1.width * d1.height
        area2 = d2.width * d2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def analyze_multiple_images(self, images: List[np.ndarray]) -> List[Dict]:
        """
        Анализ нескольких изображений для поиска дверей.

        Returns:
            Список обнаруженных дверей с агрегированной информацией
        """
        all_candidates = []

        for i, img in enumerate(images):
            detected = self.detect_doors(img)
            print(f"  Фото {i + 1}: найдено {len(detected)} кандидатов на дверь")

            for door in detected:
                all_candidates.append({
                    'photo_idx': i,
                    'door': door,
                    'relative_x': (door.x + door.width / 2) / img.shape[1],
                    'relative_y': (door.y + door.height / 2) / img.shape[0],
                    'is_valid': door.is_valid
                })

        # Группировка по позиции на стене
        by_position = {'left': [], 'right': [], 'center': []}
        for c in all_candidates:
            if c['door'].confidence > 0.3:
                by_position[c['door'].wall_position].append(c)

        final_doors = []
        for position, candidates in by_position.items():
            if len(candidates) == 0:
                continue

            # Если нашли несколько раз в разных фото - повышаем уверенность
            if len(candidates) >= 2 or (candidates and candidates[0]['door'].confidence > 0.6):
                best = max(candidates, key=lambda x: x['door'].confidence)
                confidence_boost = 1.0 + (len(candidates) - 1) * 0.15

                final_doors.append({
                    'position': position,
                    'confidence': min(best['door'].confidence * confidence_boost, 1.0),
                    'aspect_ratio': best['door'].height / best['door'].width,
                    'verified': len(candidates) >= 2,
                    'has_glass': best['door'].has_glass,
                    'is_open': best['door'].is_open
                })
                print(f"    ✓ Дверь на {position} стене: уверенность {best['door'].confidence:.2f}")

        return final_doors


def map_door_to_floorplan(detected_doors: List[Dict],
                          room_width: float,
                          room_length: float,
                          windows: List = None) -> Dict:
    """
    Преобразование обнаруженной двери в планировку комнаты.
    Возвращает одну дверь (основную).
    """
    if not detected_doors:
        return None

    # Берем дверь с максимальной уверенностью
    best_door = max(detected_doors, key=lambda d: d['confidence'])

    position = best_door['position']
    confidence = best_door['confidence']
    aspect = best_door['aspect_ratio']
    has_glass = best_door.get('has_glass', False)
    is_open = best_door.get('is_open', False)

    # Стандартные размеры двери
    door_width = 0.9
    door_height = 2.0

    # Корректировка размеров по соотношению сторон
    if aspect > 2.2:  # Очень высокая дверь
        door_height = 2.1
    elif aspect < 1.8:  # Более широкая дверь
        door_width = 1.0

    # Определяем стену и позицию
    if position == 'left':
        wall = 'left'
        # Позиция вдоль стены (от края комнаты)
        # Ставим ближе к углу, но не вплотную
        max_pos = room_length
        door_x = max_pos * 0.15

    elif position == 'right':
        wall = 'right'
        max_pos = room_length
        door_x = max_pos * 0.85 - door_width

    else:  # center - дверь на дальней стене
        wall = 'top'
        max_pos = room_width
        door_x = max_pos * 0.5 - door_width / 2

    # Если есть окна, стараемся не размещать дверь рядом с окном
    if windows:
        wall_windows = [w for w in windows if w.wall == wall]
        if wall_windows:
            # Ищем место подальше от окон
            best_gap = self._find_best_door_position(door_x, wall_windows, max_pos, door_width)
            if best_gap is not None:
                door_x = best_gap

    # Ограничиваем позицию
    door_x = max(0.1, min(door_x, max_pos - door_width - 0.1))

    print(f"\n  Дверь размещена: стена {wall}, позиция {door_x:.2f}м от края")
    if has_glass:
        print(f"  Обнаружена стеклянная дверь")
    if is_open:
        print(f"  Дверь на фото открыта")

    from room_detector import Door
    return Door(
        x=round(door_x, 2),
        width=door_width,
        height=door_height,
        wall=wall,
        has_glass=has_glass,
        is_open=is_open
    )


def _find_best_door_position(initial_x, windows, max_pos, door_width, margin=0.5):
    """Найти лучшую позицию для двери, не пересекающуюся с окнами."""

    # Проверяем, не пересекается ли с окнами
    for win in windows:
        win_start = win.x
        win_end = win.x + win.width

        door_start = initial_x
        door_end = initial_x + door_width

        # Если пересекаются, ищем альтернативу
        if not (door_end < win_start or door_start > win_end):
            # Пробуем поставить слева от окна
            left_pos = win_start - door_width - margin
            if left_pos > 0:
                return left_pos

            # Пробуем поставить справа от окна
            right_pos = win_end + margin
            if right_pos + door_width < max_pos:
                return right_pos

    return initial_x