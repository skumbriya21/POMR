import sys
import argparse
from pathlib import Path

from utils import load_images
from room_detector import RoomDimensions, Window
from floorplan import FloorplanDrawer
from window_detector import WindowDetectorCV, map_windows_to_floorplan
from model3d import create_3d_model  # НОВЫЙ ИМПОРТ


def get_room_dimensions_interactive():
    """Интерактивный ввод размеров комнаты."""
    print("ВВЕДИТЕ РАЗМЕРЫ КОМНАТЫ")

    while True:
        try:
            width_input = input("Ширина комнаты (м): ").strip().replace(',', '.')
            width = float(width_input)

            length_input = input("Длина комнаты (м): ").strip().replace(',', '.')
            length = float(length_input)

            height_input = input("Высота потолка (м) [2.7]: ").strip().replace(',', '.')
            height = float(height_input) if height_input else 2.7

            if width <= 0 or length <= 0 or height <= 0:
                print("Ошибка: размеры должны быть положительными числами!")
                continue

            area = width * length
            return RoomDimensions(width, length, height, area)

        except ValueError:
            print("Ошибка: введите числовое значение (например: 4.5)")


def get_windows_count_interactive():
    """Запрос количества окон."""
    print("ИНФОРМАЦИЯ ОБ ОКНАХ")

    while True:
        try:
            count_input = input("Сколько окон в комнате? [0]: ").strip()
            if not count_input:
                return 0
            count = int(count_input)
            if count < 0:
                print("Ошибка: количество не может быть отрицательным!")
                continue
            return count
        except ValueError:
            print("Ошибка: введите целое число!")


def get_windows_from_photos_interactive(num_images, image_paths):
    """Ручной ввод окон по фотографиям."""
    windows = []

    print(f"\nУ вас загружено {num_images} фото.")
    print("Укажите, на каких фото видны окна и на каких стенах они находятся.")
    print("\nОбозначения стен:")
    print("  1 - левая стена (left)")
    print("  2 - правая стена (right)")
    print("  3 - дальняя стена/фронт (top)")
    print("  4 - ближняя стена (bottom)")
    print("\nПример ввода: '1 на фото 1 слева, 1 на фото 2 справа'")
    print("Или пошагово:")

    while True:
        try:
            more = input("\nДобавить окно? (y/n): ").strip().lower()
            if more in ['n', 'no', 'нет', '']:
                break

            if more not in ['y', 'yes', 'да']:
                continue

            # Выбор фото
            print(f"\nДоступные фото:")
            for i, path in enumerate(image_paths, 1):
                print(f"  {i}. {Path(path).name}")

            photo_num = input("Номер фото: ").strip()
            try:
                photo_idx = int(photo_num) - 1
                if not (0 <= photo_idx < num_images):
                    print(f"Ошибка: номер фото должен быть от 1 до {num_images}")
                    continue
            except ValueError:
                print("Ошибка: введите номер фото")
                continue

            # Выбор стены
            wall_input = input("Стена (1-левая, 2-правая, 3-дальняя, 4-ближняя): ").strip()
            wall_map = {
                '1': 'left',
                '2': 'right',
                '3': 'top',
                '4': 'bottom',
                'левая': 'left',
                'правая': 'right',
                'дальняя': 'top',
                'ближняя': 'bottom',
                'left': 'left',
                'right': 'right',
                'top': 'top',
                'bottom': 'bottom'
            }

            wall = wall_map.get(wall_input.lower())
            if not wall:
                print("Ошибка: неверное обозначение стены")
                continue

            # Размеры окна (можно пропустить для типовых)
            size_input = input("Размеры окна (ширина высота в метрах) [1.2 1.5]: ").strip()
            if size_input:
                try:
                    parts = size_input.split()
                    w_width = float(parts[0])
                    w_height = float(parts[1])
                except:
                    print("Ошибка формата, используем стандартные 1.2×1.5м")
                    w_width, w_height = 1.2, 1.5
            else:
                w_width, w_height = 1.2, 1.5

            # Позиция вдоль стены
            pos_input = input("Позиция вдоль стены (0.0-1.0, где 0.5 - середина) [0.5]: ").strip()
            try:
                position_ratio = float(pos_input) if pos_input else 0.5
                position_ratio = max(0.0, min(1.0, position_ratio))
            except:
                position_ratio = 0.5

            windows.append({
                'photo_idx': photo_idx,
                'wall': wall,
                'width': w_width,
                'height': w_height,
                'position_ratio': position_ratio,
                'manual': True
            })

            print(f"  ✓ Окно добавлено: {w_width}м × {w_height}м, стена {wall}")

        except KeyboardInterrupt:
            print("\nПрервано пользователем")
            break

    return windows


def combine_detected_and_manual(detected_windows, manual_windows, room_width, room_length):
    """Комбинирование автоматически найденных и ручно добавленных окон."""
    result = []
    used_walls = set()

    # Сначала добавляем подтверждённые автоматические
    for win in detected_windows:
        if win.get('verified') or win['confidence'] > 0.6:
            result.append(win)
            used_walls.add(win['position'])
            print(f"  ✓ Авто-окно подтверждено: стена {win['position']}")

    # Добавляем ручные, проверяя конфликты
    for mwin in manual_windows:
        wall = mwin['wall']

        # Если на этой стене уже есть окно — спрашиваем
        if wall in used_walls:
            print(f"  На стене {wall} уже есть окно из авто-детекции")
            replace = input(f"    Заменить ручным? (y/n): ").strip().lower()
            if replace not in ['y', 'yes', 'да']:
                continue
            # Удаляем старое
            result = [w for w in result if w.get('wall') != wall and w.get('position') != wall]

        # Конвертируем в формат для планировки
        if wall in ['left', 'right']:
            max_pos = room_length
        else:
            max_pos = room_width

        x = max_pos * mwin['position_ratio']

        result.append({
            'position': wall,
            'wall': wall,
            'confidence': 1.0,
            'aspect_ratio': mwin['height'] / mwin['width'],
            'verified': True,
            'x': x,
            'y': 1.0,
            'width': mwin['width'],
            'height': mwin['height']
        })

        used_walls.add(wall)
        print(f"  ✓ Ручное окно: {mwin['width']}м × {mwin['height']}м, стена {wall}")

    return result


def main():
    parser = argparse.ArgumentParser(description='Создание планировки и 3D модели комнаты по фото')
    parser.add_argument('--images', '-i', nargs='+', help='Пути к фотографиям')
    parser.add_argument('--output', '-o', default='floorplan.png', help='Выходной файл планировки')
    parser.add_argument('--output-3d', default='room.obj', help='Выходной файл 3D модели (.obj)')
    parser.add_argument('--width', '-w', type=float, help='Ширина комнаты (м)')
    parser.add_argument('--length', '-l', type=float, help='Длина комнаты (м)')
    parser.add_argument('--height', '-H', type=float, default=2.7, help='Высота потолка (м)')
    parser.add_argument('--auto-only', action='store_true',
                        help='Только автоматическая детекция без подтверждения')
    parser.add_argument('--manual-only', action='store_true',
                        help='Только ручной ввод окон')
    parser.add_argument('--no-3d', action='store_true',
                        help='Не создавать 3D модель')

    args = parser.parse_args()

    print("  RoomPlanner - Создание планировки и 3D модели по фото")

    # === Получаем пути к фото ===
    if args.images:
        image_paths = args.images
    else:
        print("\nВведите пути к фотографиям (через пробел):")
        print("Пример: D:\\img1.jpg D:\\img2.jpg D:\\img3.jpg D:\\img4.jpg")
        user_input = input("> ").strip()
        if not user_input:
            print("Ошибка: не указаны фотографии!")
            sys.exit(1)
        image_paths = user_input.split()

    # Проверка файлов
    for path in image_paths:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден: {path}")
            sys.exit(1)

    print(f"\nЗагрузка {len(image_paths)} изображений...")
    images = load_images(image_paths)

    if len(images) == 0:
        print("Ошибка: Не удалось загрузить изображения!")
        sys.exit(1)

    # === ЭТАП 1: Определение размеров комнаты ===
    print("\n[1/3] Определение размеров комнаты...")

    if args.width and args.length:
        room_dims = RoomDimensions(
            width=args.width,
            length=args.length,
            height=args.height,
            area=args.width * args.length
        )
        print(f"  ✓ Размеры из командной строки: {room_dims.width}м × {room_dims.length}м")
    else:
        room_dims = get_room_dimensions_interactive()
        print(f"  ✓ Размеры комнаты: {room_dims.width}м × {room_dims.length}м")

    # === ЭТАП 2: Анализ и подтверждение окон ===
    print("\n[2/3] Анализ фотографий на наличие окон...")

    detected_windows = []
    manual_windows = []

    # Автоматическая детекция (если не --manual-only)
    if not args.manual_only:
        window_detector = WindowDetectorCV()
        detected_windows = window_detector.analyze_multiple_images(images)
        print(f"\n  Автоматически обнаружено: {len(detected_windows)} окон")
        for w in detected_windows:
            status = "✓" if w.get('verified') else "~"
            print(f"    {status} Стена {w['position']}: уверенность {w['confidence']:.2f}")

    # Ручной ввод или подтверждение
    if not args.auto_only:
        expected_count = get_windows_count_interactive()

        if expected_count == 0:
            print("  → Продолжаем без окон")
            final_windows = []
        else:
            current_count = len(detected_windows)

            if current_count < expected_count or args.manual_only:
                print(f"\n  Найдено {current_count}, нужно {expected_count}")
                print("  Переходим к ручному указанию окон...")

                manual_windows = get_windows_from_photos_interactive(len(images), image_paths)

                # Комбинируем
                final_windows = combine_detected_and_manual(
                    detected_windows, manual_windows, room_dims.width, room_dims.length
                )
            else:
                # Подтверждаем автоматические
                if detected_windows and not args.manual_only:
                    print(f"\n  Найдено {len(detected_windows)} окон.")
                    confirm = input("  Использовать все? (y/n/редактировать): ").strip().lower()

                    if confirm in ['n', 'no', 'нет']:
                        # Убираем сомнительные
                        final_windows = [w for w in detected_windows if w['confidence'] > 0.7]
                        print(f"  Оставлено {len(final_windows)} (только высокая уверенность)")
                    elif confirm in ['edit', 'редактировать', 'ред']:
                        # Редактирование
                        final_windows = []
                        for i, w in enumerate(detected_windows, 1):
                            keep = input(f"  Окно {i} на стене {w['position']}? (y/n) [y]: ").strip().lower()
                            if keep not in ['n', 'no', 'нет']:
                                final_windows.append(w)
                    else:
                        final_windows = detected_windows
                else:
                    final_windows = detected_windows

            # Проверяем итоговое количество
            if len(final_windows) < expected_count:
                print(f"\n  ⚠ Внимание: указано {expected_count} окон, но на плане {len(final_windows)}")
                add_more = input("  Добавить недостающие? (y/n): ").strip().lower()
                if add_more in ['y', 'yes', 'да']:
                    additional = get_windows_from_photos_interactive(len(images), image_paths)
                    final_windows = combine_detected_and_manual(
                        final_windows, additional, room_dims.width, room_dims.length
                    )
    else:
        # Только авто
        final_windows = detected_windows

    print(f"\n  ✓ Итого окон для планировки: {len(final_windows)}")

    # === ЭТАП 3: Создание планировки ===
    print("\n[3/3] Создание планировки...")

    # Конвертируем в формат для отрисовки
    windows_for_drawer = []
    windows_3d = []  # Для 3D модели

    for w in final_windows:
        # Если уже есть x/y (из ручного ввода)
        if 'x' in w and 'y' in w:
            win_obj = Window(
                x=w['x'],
                y=w['y'],
                width=w['width'],
                height=w['height'],
                wall=w['wall']
            )
        else:
            # Из авто-детекции — через map_windows_to_floorplan
            mapped = map_windows_to_floorplan([w], room_dims.width, room_dims.length)
            if mapped:
                m = mapped[0]
                win_obj = Window(
                    x=m['x'],
                    y=m['y'],
                    width=m['width'],
                    height=m['height'],
                    wall=m['wall']
                )
            else:
                continue

        windows_for_drawer.append(win_obj)
        windows_3d.append(win_obj)

    drawer = FloorplanDrawer()
    drawer.draw(room_dims, windows_for_drawer, args.output)

    # === ЭТАП 4: Создание 3D модели ===
    if not args.no_3d:
        create_3d_model(room_dims, windows_3d, args.output_3d)

    # Итог
    print("ГОТОВО!")
    print(f"Размеры комнаты: {room_dims.width}м × {room_dims.length}м = {room_dims.area}м²")
    print(f"Высота потолка: {room_dims.height}м")
    print(f"Окон: {len(windows_3d)}")
    print(f"\nФайлы:")
    print(f"  Планировка: {args.output}")
    if not args.no_3d:
        print(f"  3D модель:  {args.output_3d}")
        print(f"  Материалы:  {args.output_3d.replace('.obj', '.mtl')}")


if __name__ == "__main__":
    main()