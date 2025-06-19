# import cv2
# import os
# import pytesseract
# import numpy as np
# from PIL import Image, ImageDraw  # Установите Pillow: pip install Pillow

# cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
# nPlateCascade = cv2.CascadeClassifier(cascade_path)
# minArea = 200
# color = (255, 0, 255)

# cap = cv2.VideoCapture("data/videos/9.mp4")
# if cap is None or not cap.isOpened():
#     print("Video not found or cannot open. Please check the path.")
#     exit()

# width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
# height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
# scale = 1

# cap.set(3, width)
# cap.set(4, height)
# cap.set(10, 150)
# count = 0
# recognized_plates = {}

# while True:
#     success, img = cap.read()
#     if not success:
#         break
#     img = cv2.flip(img, -1)
#     img = cv2.resize(img, (int(width * scale), int(height * scale)))
#     imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
#     for (x, y, w, h) in numberPlates:
#         area = w * h
#         if area > minArea:
#             cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
#             cv2.putText(img, "Number Plate", (x, y - 5),
#                         cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color, 2)
#             imgRoi = img[y:y + h, x:x + w]

#             # Предобработка ROI
#             imgGrayRoi = cv2.cvtColor(imgRoi, cv2.COLOR_BGR2GRAY)
#             # Адаптивная пороговая обработка
#             imgThresh = cv2.adaptiveThreshold(imgGrayRoi, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

#             # Размытие (опционально)
#             # imgThresh = cv2.medianBlur(imgThresh, 3)

#             # Морфологические операции (эрозия и дилатация)
#             kernel = np.ones((2, 2), np.uint8)
#             imgErode = cv2.erode(imgThresh, kernel, iterations=1)
#             imgDilate = cv2.dilate(imgErode, kernel, iterations=1)
#             roi_pil = Image.fromarray(imgDilate)


#             # Используйте imgDilate для распознавания (или imgErode, если лучше)
#             cv2.imshow("ROI", imgRoi)
#             cv2.imshow("ROI_Threshold", imgDilate)  # Отображаем обработанное изображение

#     # cv2.imshow("Result", img)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('s'):
#         if 'imgRoi' in locals():
#             filename = f"NoPlate_{count}.jpg"
#             save_path = os.path.join("Resources/Scanned", filename)
#             cv2.imwrite(save_path, imgDilate)  # Сохраняем обработанное ROI

#             # Распознавание номера
#             try:
#                 plate_text = pytesseract.image_to_string(imgDilate, lang='rus', config='--psm 8 --oem 13')
#                 plate_text = ''.join(filter(str.isalnum, plate_text))
#                 recognized_plates[filename] = plate_text
#                 print(f"Saved: {filename} | Text: {plate_text}")
#             except Exception as e:
#                 print(f"Ошибка распознавания: {e}")
#                 plate_text = "Error"
#                 recognized_plates[filename] = plate_text

#             # Отображение текста на изображении
#             img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
#             draw = ImageDraw.Draw(img_pil)
#             draw.text((x, y - 20), plate_text, fill=(0, 255, 0))
#             img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
#             # cv2.imshow("Result", img)

#             cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
#             cv2.putText(img, "Scan Saved", (150, 265), cv2.FONT_HERSHEY_DUPLEX,
#                         2, (0, 0, 255), 2)
#             # cv2.imshow("Result", img)
#             cv2.waitKey(500)
#             count += 1
#         else:
#             print("No number plate detected to save.")
#     elif key == ord('q'):
#         print("Exiting...")
#         break

# cap.release()
# cv2.destroyAllWindows()

# print("\n📄 Распознанные номера:")
# for file, number in recognized_plates.items():
#     print(f"{file}: {number}")

# test for pytesseract


import pytesseract
from PIL import Image  # pip install Pillow

try:
    image = Image.open("data/images/5.jpg")  # Замените "test.png" на путь к вашему тестовому изображению
    text = pytesseract.image_to_string(image, lang='rus')
    print(text)
except Exception as e:
    print(f"Error: {e}")


"""
pytesseract on russian work correct
but he cant to understand detected numbers, i must use Networks

Привет. Это образец для распознавания текста для статьи в Т—Ж.
Проверим, сколько слов он определит полностью, сколько
превратит в набор букв и сколько вообще не узнает:

1. Набор слов для распознавания № 1.
2. Еще один набор слов.

3. Третий набор слово.

4. Экспрессия -— четвертое слово.

5. Финальное пятое слово.
"""

