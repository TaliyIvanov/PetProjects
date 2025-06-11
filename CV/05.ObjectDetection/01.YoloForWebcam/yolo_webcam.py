from ultralytics import YOLO
import cv2
import cvzone
import math

# webcamera
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # weight
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # height
cap.set(cv2.CAP_PROP_FPS, 30)  # framerate

# model
model = YOLO('models/yolov8n.pt')

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

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # input our video in model
    results = model(frame)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # bounding box
            x1, y1, x2, y2 = list(map(int, box.xyxy[0])) # x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            # this for opencv
            # print(x1, y1, x2, y2)
            # cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)

            # this for cvzone
            bbox = x1, y1, x2-x1, y2-y1
            cvzone.cornerRect(frame, bbox)
            # Confidence
            conf = math.ceil(box.conf[0].item()*100)/100
            # Classname
            cls = int(box.cls[0])
            print()
            cvzone.putTextRect(frame, f'{classNames[cls]} {conf}', (max(0,x1), max(35,y1)), scale=1, thickness=1)



    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()