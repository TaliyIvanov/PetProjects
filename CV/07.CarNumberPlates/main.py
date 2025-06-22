# imports
import cv2
from ultralytics import YOLO
import numpy as np
import sort as s
import easyocr
import os

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

    # detect the number plate with opencv
    numberPlates = nPlateCascade.detectMultiScale(gray, 1.1, 10)
    for (x,y,w,h) in numberPlates:
        area = w * h
        if area > minarea:
            cv2.rectangle(frame, (x,y), (x+w, y+h), numbercolor, 1)
            cv2.putText(frame, 'NumberPlate', (x, y-5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, numbercolor, 1)
            frame_number = frame[y:y+h, x:x+w, :]
            
            cv2.imshow('Number', frame_number) # show numberplate

            # prepair numberplate for easyOCR

    cv2.imshow('frame', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print('Exiting...')
        break

cv2.destroyAllWindows()





# OCR the number text with easyOCR

# write file.txt

