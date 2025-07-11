# Imports
from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sort as s

# Stream
cap = cv2.VideoCapture('data/videos/3.mp4')
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()

# Video size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Mask
mask = cv2.imread('data/images/mask_for_3_video.jpg')
if mask is None:
    print("Mask image not found. Please check the path.")

# Model
model = YOLO('models/yolov8n.pt')

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

# Tracking
tracker = s.Sort(max_age=20, min_hits=3, iou_threshold=0.3)
total_counts = []
track_history = {}

# Line-counter
limits = [400, 475, 1200, 475]
pt1, pt2 = (limits[0], limits[1]), (limits[2], limits[3])

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Resize if need
    if width != 1280 or height != 720:
        frame = cv2.resize(frame, (1280, 720))

    frame_region = cv2.bitwise_and(frame, mask)
    results = model(frame_region, stream=True)

    detections = np.empty((0, 5))
    currentNames = ['car', 'truck', 'bus', 'motorbike']

    # Detection 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = box.conf[0].item()
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass in currentNames and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Tracking
    resultsTracker = tracker.update(detections)
    cv2.line(frame, pt1, pt2, (0, 0, 255), 3) # Show our line

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        # Checking the line crossing
        if id in track_history:
            prev_cx, prev_cy = track_history[id]
            if prev_cy < limits[1]+5 and cy >= limits[1]-5:  # пересечение снизу вверх
                if id not in total_counts:
                    total_counts.append(id)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)
        
        # Update history
        track_history[id] = (cx, cy)

        # Visualization
        cvzone.putTextRect(frame, f'{id}', (x1, y1 - 10), scale=1.5, thickness=2)
        cvzone.cornerRect(frame, (x1, y1, x2 - x1, y2 - y1), rt=2)
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

    # Show counter
    cvzone.putTextRect(frame, f'Counts: {len(total_counts)}', (50, 50), scale=2, thickness=3)

    cv2.imshow("frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
