import cv2

# Путь к исходному видео
cap = cv2.VideoCapture('data/my_phone_video_from_street.mp4')
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()

# Желаемый размер окна вывода
new_width = 1280
new_height = 720

# Получаем исходные размеры видео
original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(original_width)
print(original_height)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Выводим информацию о кадре
    if frame is not None:
        print("Кадр:")
        print("  Форма (height, width, channels):", frame.shape)
        print("  Тип данных:", frame.dtype)  # Например, uint8
        print("  Размер:", frame.size)  # Общее количество элементов (пикселей)

    # # Обрезка кадра
    # if original_width > new_width or original_height > new_height:
    #     # Вычисляем координаты обрезки, чтобы обрезать по центруq
    #     x1 = (original_width - new_width) // 2
    #     y1 = (original_height - new_height) // 2
    #     x2 = x1 + new_width
    #     y2 = y1 + new_height

    #     # Обрезаем кадр
    #     cropped_frame = frame[:,:]
    # else:
    #     cropped_frame = frame  # Если исходный размер меньше целевого, просто используем исходный кадр

    # # Отображаем обрезанный кадр
    # cv2.imshow('Cropped Frame', cropped_frame)
    cv2.imshow('Frame', frame[600:1320,:])
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()