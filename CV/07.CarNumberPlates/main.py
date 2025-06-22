# imports
import cv2
from ultralytics import YOLO
import numpy as np
import sort as s
import easyocr

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
#         breakS
    
cv2.destroyAllWindows()

# load model


# pipeline
while True:
    success, frame = cap.read()

# detect car with Yolo8n


# detect the number plate with opencv

# OCR the number text with easyOCR

# write file.txt

