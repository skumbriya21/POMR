import cv2
import numpy as np
from typing import List, Dict
from dataclasses import dataclass


@dataclass
class DetectedWindow:
    x: int
    y: int
    width: int
    height: int
    confidence: float
    wall_position: str
    is_valid: bool = True


class WindowDetectorCV:
    def __init__(self, min_area=5000, max_area=500000):
        self.min_area = min_area
        self.max_area = max_area

        self.typical_window = {
            'aspect_ratio': (0.8, 2.5),
            'relative_height': (0.3, 0.7),
            'wall_margin': 0.15
        }

    def detect_windows(self, image: np.ndarray) -> List[DetectedWindow]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        windows = []
        img_height, img_width = image.shape[:2]

        brightness = np.mean(gray)
        _, bright_mask = cv2.threshold(gray, brightness * 1.2, 255, cv2.THRESH_BINARY)

        for contour in contours:
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            if len(approx) != 4:
                continue

            x, y, w, h = cv2.boundingRect(approx)
            area = w * h

            if not (self.min_area < area < self.max_area):
                continue

            aspect = h / float(w)
            if not (0.5 < aspect < 3.0):
                continue

            rect_area = w * h
            contour_area = cv2.contourArea(contour)
            fill_ratio = contour_area / rect_area if rect_area > 0 else 0

            if fill_ratio > 0.95 and area > 50000:
                continue

            center_y = y + h / 2
            relative_y = center_y / img_height

            if relative_y > 0.85 or relative_y < 0.1:
                continue

            roi = bright_mask[y:y + h, x:x + w]
            brightness_ratio = np.sum(roi > 0) / (w * h) if w * h > 0 else 0

            center_x = x + w / 2
            if center_x < img_width * 0.3:
                position = 'left'
            elif center_x > img_width * 0.7:
                position = 'right'
            else:
                position = 'center'

            confidence = self._calculate_confidence(aspect, fill_ratio, brightness_ratio, area)
            is_valid = confidence > 0.4 and brightness_ratio > 0.15

            windows.append(DetectedWindow(
                x=x, y=y, width=w, height=h,
                confidence=confidence,
                wall_position=position,
                is_valid=is_valid
            ))

        windows.sort(key=lambda w: w.confidence, reverse=True)
        valid_windows = [w for w in windows if w.is_valid]

        if len(valid_windows) == 0 and len(windows) > 0:
            best = windows[0]
            best.confidence *= 0.5
            valid_windows = [best]

        return self._remove_overlapping(valid_windows)

    def _calculate_confidence(self, aspect, fill_ratio, brightness, area):
        aspect_score = 1.0 - abs(aspect - 1.5) / 1.5
        aspect_score = max(0, aspect_score)

        fill_score = 1.0 - abs(fill_ratio - 0.85) / 0.3 if 0.5 < fill_ratio < 1.0 else 0
        bright_score = min(brightness * 2, 1.0)
        size_score = min(area / 50000, 1.0) * (1.0 if area < 200000 else 0.7)

        return (aspect_score * 0.3 + fill_score * 0.2 + bright_score * 0.3 + size_score * 0.2)

    def _remove_overlapping(self, windows: List[DetectedWindow], iou_threshold=0.3):
        if len(windows) <= 1:
            return windows

        keep = []
        for w1 in windows:
            overlap = False
            for w2 in keep:
                if self._iou(w1, w2) > iou_threshold:
                    overlap = True
                    break
            if not overlap:
                keep.append(w1)
        return keep

    def _iou(self, w1, w2) -> float:
        x1 = max(w1.x, w2.x)
        y1 = max(w1.y, w2.y)
        x2 = min(w1.x + w1.width, w2.x + w2.width)
        y2 = min(w1.y + w1.height, w2.y + w2.height)

        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = w1.width * w1.height
        area2 = w2.width * w2.height
        union = area1 + area2 - intersection

        return intersection / union if union > 0 else 0

    def analyze_multiple_images(self, images: List[np.ndarray]) -> List[Dict]:
        all_candidates = []

        for i, img in enumerate(images):
            detected = self.detect_windows(img)
            print(f"  Фото {i + 1}: найдено {len(detected)} кандидатов")

            for win in detected:
                all_candidates.append({
                    'photo_idx': i,
                    'window': win,
                    'relative_x': (win.x + win.width / 2) / img.shape[1],
                    'relative_y': (win.y + win.height / 2) / img.shape[0],
                    'is_valid': win.is_valid
                })

        by_position = {'left': [], 'right': [], 'center': []}
        for c in all_candidates:
            if c['window'].confidence > 0.3:
                by_position[c['window'].wall_position].append(c)

        final_windows = []
        for position, candidates in by_position.items():
            if len(candidates) == 0:
                continue

            if len(candidates) >= 2 or candidates[0]['window'].confidence > 0.7:
                best = max(candidates, key=lambda x: x['window'].confidence)
                confidence_boost = 1.0 + (len(candidates) - 1) * 0.1

                final_windows.append({
                    'position': position,
                    'confidence': min(best['window'].confidence * confidence_boost, 1.0),
                    'aspect_ratio': best['window'].height / best['window'].width,
                    'verified': len(candidates) >= 2
                })
                print(f"    ✓ Окно на {position} стене: уверенность {best['window'].confidence:.2f}")

        return final_windows


def map_windows_to_floorplan(detected_windows: List[Dict],
                             room_width: float,
                             room_length: float) -> List[Dict]:
    """
    Преобразование обнаруженных окон в планировку комнаты.
    """
    floorplan_windows = []

    for win in detected_windows:
        position = win['position']
        confidence = win['confidence']
        aspect = win['aspect_ratio']

        if aspect > 1.5:
            w_width = 0.8
            w_height = 1.5
        else:
            w_width = 1.5
            w_height = 1.2

        if position == 'left':
            wall = 'left'
            x = room_length * 0.3
        elif position == 'right':
            wall = 'right'
            x = room_length * 0.3
        else:  # center
            wall = 'top'
            x = room_width * 0.4

        floorplan_windows.append({
            'x': x,
            'y': 1.0,
            'width': w_width,
            'height': w_height,
            'wall': wall,
            'confidence': confidence
        })

    print(f"\nПеренесено на планировку: {len(floorplan_windows)} окон")
    for i, w in enumerate(floorplan_windows, 1):
        print(f"  Окно {i}: {w['width']}м × {w['height']}м, стена {w['wall']}")

    return floorplan_windows