# imports
from ultralytics import YOLO
import cv2
import numpy as np
from util import get_car
import sort as s

tracker = s.Sort()

# load models
model_nano = YOLO('models/yolov8n.pt')
license_plate_detector = YOLO('models/license_plate_detector.pt')

# Classes
classNames = [
    "person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
    "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
    "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
    "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush"
]

vehicles = [3,5,7] # ["car", "bus", "truck"]


# load video
# cap = cv2.VideoCapture('data/my_phone_video_from_street.mp4')
cap = cv2.VideoCapture('data/test_video.mp4')
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()

# read frames
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # detect vehicles
    detections = model_nano(frame)[0]
    detected = []
    for detection in detections.boxes.data.tolist():
        # print(detection)
        # [157.6859130859375, 195.745849609375, 249.51446533203125, 272.1181640625, 0.7862106561660767, 2.0]
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles: # ["car", "bus", "truck"]
            detected.append([x1, y1, x2, y2, score])
            
 
    # track vehicles
    track_ids = tracker.update(np.asarray(detected))

    # detect license plate to car
    detections_plates = license_plate_detector(frame)[0]
    for license_plate in detections_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # assign license plate
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)

        # crop license plate
        license_plate_crop = frame[int(y1):int(y2), int(x1):int(x2), :]

        # process license plate
        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
        _, license_plate_crop_thresh = cv2.threshold(license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV)
        
        cv2.imshow('crop', license_plate_crop)
        cv2.imshow('thresh', license_plate_crop_thresh)
        # read license plate number


        # write results

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == (ord('q')):
        break

cap.release()
cv2.destroyAllWindows()