# imports
import cv2
from ultralytics import YOLO
import numpy as np
import sort as s
import easyocr
import os
import math

from utils import warp_perspective, correct_common_ocr_errors, clean_plate, correct_number, format_license

# create numbers plates
cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
nPlateCascade = cv2.CascadeClassifier(cascade_path)
numbercolor = (0, 0, 255)
minarea = 200
scale = 0.5

# load video
cap = cv2.VideoCapture('data/test_video.mp4')
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()
# check video without any tranforms (before do anything)
# while True:
#     success, frame = cap.read()
#     if not success:
#         print("No frame is read")
#         break # Exit the loop if no frame is read

#     cv2.imshow('Frame', frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         print('Exiting...')
#         break
# cv2.destroyAllWindows()

# load model
model = YOLO('models/yolov8n.pt')

# load number reader
number_reader = easyocr.Reader(['ru'], gpu=True)
count = 0
number_dict = {}


# pipeline
while True:
    success, frame = cap.read()
    if not success:
        break
    # prepair image
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    frame = frame[400:1520,:,:]
    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (int(width * scale), int(height * scale)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect car with Yolo8n
    detected_cars = []
    car_detection = model(frame, verbose=False)[0] # verbose=False
    for detection in car_detection.boxes.data.tolist():
        # print(detection)
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) == 2: # car
            detected_cars.append([x1, y1, x2, y2, score])
            conf = math.ceil(score*100)/100
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 1)
            cv2.putText(frame, f'Car, {conf}', (int(x1), int(y1)-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)

    # detect the number plate with opencv
    numberPlates = nPlateCascade.detectMultiScale(gray, 1.1, 10)
    for (x,y,w,h) in numberPlates:
        area = w * h
        if area > minarea:
            cv2.rectangle(frame, (x,y), (x+w, y+h), numbercolor, 1)
            cv2.putText(frame, 'NumberPlate', (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, numbercolor, 1)
            pts = [(x,y),(x+w,y), (x+w,y+h), (x,y+h)]
            # frame_number = frame[y:y+h, x:x+w, :]
            # correction perspective
            frame_number = warp_perspective(frame, pts)
            cv2.imshow('Number', frame_number) # show numberplate

            # prepair numberplate for easyOCR
            # height_number, width_number = frame_number.shape[:2]
            # frame_number = cv2.resize(frame_number, (int(width_number * 2), int(height_number * 2)))
            frame_number_gray = cv2.cvtColor(frame_number, cv2.COLOR_BGR2GRAY)
            frame_number_prep = cv2.adaptiveThreshold(frame_number_gray,
                                                         255,
                                                         cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                         cv2.THRESH_BINARY,
                                                         11,
                                                         2)
            # show prepaire number
            cv2.imshow('Prepaired number', frame_number_prep) # show numberplate

            # OCR the number text with easyOCR
            readed_numbers = number_reader.readtext(frame_number)
            for (bbox, text, prob) in readed_numbers:
                if len(text) > 5:
                    text = text.upper()
                    text = correct_common_ocr_errors(text)
                    text = clean_plate(text)

                    if not text:
                        continue  # Пропускаем, если вернулось None
                    
                    if not correct_number(text):
                        continue
                    text = format_license(text)
                    print(f'Number: {text} Score: {math.ceil((prob*100))/100}')
                    number_dict[count] = text
                    count += 1



    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exiting...')
        break

cv2.destroyAllWindows()







# write file.txt

