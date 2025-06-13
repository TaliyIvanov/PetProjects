from ultralytics import YOLO
import cv2
import cvzone
import math
import numpy as np
import sort as s

# Stream
cap = cv2.VideoCapture('data/videos/8.mp4')
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()

# Video size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

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
track_history = {}  # {id: (prev_cx, prev_cy)}

# Line-conter
limits = [0, 360, 1280, 360]
pt1 = (limits[0], limits[1])
pt2 = (limits[2], limits[3])

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Resize if need
    if width != 1280 or height != 720:
        frame = cv2.resize(frame, (1280, 720))

    results = model(frame)
    detections = np.empty((0, 5))

    # Detection 
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            w, h = x2 - x1, y2 - y1
            conf = math.ceil(box.conf[0].item() * 100) / 100
            cls = int(box.cls[0])
            currentClass = classNames[cls]

            if currentClass == 'person' and conf > 0.3:
                currentArray = np.array([x1, y1, x2, y2, conf])
                detections = np.vstack((detections, currentArray))

    # Tracking
    resultsTracker = tracker.update(detections)

    # Show our line
    cv2.line(frame, pt1, pt2, (0, 0, 255), 3)

    current_ids = set()

    for result in resultsTracker:
        x1, y1, x2, y2, id = map(int, result)
        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        current_ids.add(id)

        # Visualization
        cvzone.putTextRect(frame, f'{id}', (max(0, x1), max(35, y1)), scale=2, thickness=3, offset=5)
        cvzone.cornerRect(frame, (x1, y1, w, h), rt=2, colorR=(255, 0, 255))
        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)

        # Checking the line crossing
        if id in track_history:
            prev_cx, prev_cy = track_history[id]
            if prev_cy > limits[1] and cy <= limits[1]:
                if id not in total_counts:
                    total_counts.append(id)
                    cv2.line(frame, pt1, pt2, (0, 255, 0), 3)

        # Update history
        track_history[id] = (cx, cy)

    # Deleting disappeared IDs
    ids_to_remove = [tid for tid in track_history if tid not in current_ids]
    for tid in ids_to_remove:
        del track_history[tid]

    # Output counter
    cvzone.putTextRect(frame, f'Counts: {len(total_counts)}', (50, 50), scale=2, thickness=2, offset=5)

    # Show
    cv2.imshow('People Counter', frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()