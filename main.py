import sys
import argparse
from pathlib import Path

from utils import load_images, estimate_camera_matrix
from reconstruction import RoomReconstructor
from room_detector import RoomDetector, ask_room_dimensions, RoomDimensions, Window
from floorplan import FloorplanDrawer


def ask_windows_manual(room_width, room_length):
    """Ручной ввод окон."""
    windows = []

    print("ДОБАВЛЕНИЕ ОКОН")

    print("Доступные стены:")
    print(" 1 - Левая (левая сторона при входе)")
    print(" 2 - Правая (правая сторона при входе)")
    print(" 3 - Дальняя (стена напротив входа)")
    print(" 4 - Ближняя (стена у входа)")

    while True:
        add = input("\nДобавить окно? (да/нет): ").strip().lower()
        if add not in ['да', 'д', 'yes', 'y']:
            break

        try:
            wall_choice = input("Стена (1-4): ").strip()
            wall_map = {'1': 'left', '2': 'right', '3': 'top', '4': 'bottom'}
            wall = wall_map.get(wall_choice, 'right')

            w_width = float(input("Ширина окна (м): "))
            w_height = float(input("Высота окна (м): "))

            # Позиция вдоль стены
            if wall in ['left', 'right']:
                max_pos = room_length
                prompt = f"Расстояние от дальнего угла вдоль стены (0-{max_pos}м): "
            else:
                max_pos = room_width
                prompt = f"Расстояние от левого угла вдоль стены (0-{max_pos}м): "

            x = float(input(prompt))
            y = float(input("Высота подоконника от пола (м) [1.0]: ") or "1.0")

            windows.append(Window(x=x, y=y, width=w_width, height=w_height, wall=wall))
            print(f"Окно добавлено: {w_width}м × {w_height}м")

        except ValueError:
            print("Ошибка ввода! Пропускаем.")

    return windows


def main():
    parser = argparse.ArgumentParser(description='3D реконструкция комнаты')
    parser.add_argument('--images', nargs='+', help='Пути к фотографиям')
    parser.add_argument('--output', '-o', default='floorplan.png', help='Выходной файл')
    parser.add_argument('--width', type=float, help='Ширина комнаты (м)')
    parser.add_argument('--length', type=float, help='Длина комнаты (м)')

    args = parser.parse_args()

    print("  RoomReconstarction - Реконструкция комнаты")

    # Если размеры указаны в командной строке
    if args.width and args.length:
        room_dims = RoomDimensions(
            width=args.width,
            length=args.length,
            height=2.7,
            area=args.width * args.length
        )
        print(f"\nИспользуются заданные размеры: {room_dims.width}м × {room_dims.length}м")

        # Спрашиваем окна
        windows = ask_windows_manual(room_dims.width, room_dims.length)

        # Рисуем
        drawer = FloorplanDrawer()
        drawer.draw(room_dims, windows, args.output)
        print("\nГотово!")
        return

    # Если есть изображения
    if args.images:
        image_paths = args.images
    else:
        # Интерактивный ввод
        print("\nВведите пути к фотографиям (минимум 2, через пробел):")
        print("Пример: D:\\img1.jpg D:\\img2.jpg D:\\img3.jpg")
        user_input = input("> ").strip()
        image_paths = user_input.split()

    if len(image_paths) < 2:
        print("Ошибка: Нужно минимум 2 фотографии!")
        sys.exit(1)

    # Проверка файлов
    for path in image_paths:
        if not Path(path).exists():
            print(f"Ошибка: Файл не найден: {path}")
            sys.exit(1)

    print(f"\nЗагрузка {len(image_paths)} изображений...")
    images = load_images(image_paths)

    if len(images) < 2:
        print("Ошибка: Не удалось загрузить изображения!")
        sys.exit(1)

    # Спрашиваем размеры
    manual_dims = ask_room_dimensions()

    if manual_dims is not None:
        # Используем введенные размеры
        room_dims = manual_dims
        print(f"\nИспользуются введенные размеры: {room_dims.width}м × {room_dims.length}м")

        # Спрашиваем окна вручную
        windows = ask_windows_manual(room_dims.width, room_dims.length)

        # Рисуем
        drawer = FloorplanDrawer()
        drawer.draw(room_dims, windows, args.output)

    else:
        # Автоопределение из фотографий
        print("\nЗапуск 3D реконструкции...")

        # Оценка матрицы камеры
        K = estimate_camera_matrix(images[0].shape)

        # Реконструкция
        try:
            reconstructor = RoomReconstructor(K)
            points_3d, cameras = reconstructor.reconstruct(images)

            # Определение размеров
            print("\nОпределение размеров комнаты...")
            detector = RoomDetector()
            room_dims = detector.detect_room(points_3d)

            print(f"\n{'='*50}")
            print("РЕЗУЛЬТАТЫ:")
            print(f"{'='*50}")
            print(f"Ширина:  {room_dims.width} м")
            print(f"Длина:   {room_dims.length} м")
            print(f"Высота:  {room_dims.height} м")
            print(f"Площадь: {room_dims.area} м²")
            print(f"{'='*50}")

            # Поиск окон
            windows = detector.detect_windows(points_3d, images)
            print(f"\nНайдено окон: {len(windows)}")

            # Рисование планировки
            print("\nСоздание планировки...")
            drawer = FloorplanDrawer()
            drawer.draw(room_dims, windows, args.output)

        except Exception as e:
            print(f"\nОшибка реконструкции: {e}")
            print("\nПереключаемся на ручной ввод...")

            # Fallback на ручной ввод
            room_dims = ask_room_dimensions()
            if room_dims is None:
                room_dims = RoomDimensions(width=5.0, length=4.0, height=2.7, area=20.0)

            windows = ask_windows_manual(room_dims.width, room_dims.length)

            drawer = FloorplanDrawer()
            drawer.draw(room_dims, windows, args.output)

    print("\nГотово!")


if __name__ == "__main__":
    main()

