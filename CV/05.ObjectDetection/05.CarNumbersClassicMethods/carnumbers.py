import cv2 
import os
import numpy as np
import easyocr


cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_russian_plate_number.xml")
nPlateCascade = cv2.CascadeClassifier(cascade_path)
minArea = 200
color = (255,0,255)

cap = cv2.VideoCapture("data/videos/9.mp4")
if cap is None or not cap.isOpened():
    print("Video not found or cannot open. Please check the path.")
    exit()

# Video size
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
scale = 0.5

cap.set(3, width)
cap.set(4, height)
cap.set(10,150)
count = 0
number_dict = {}

reader = easyocr.Reader(['ru'], gpu=True)

# detecting number
while True:
    success, img = cap.read()
    if not success:
        break  # Exit the loop if no frame is read
    img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    rotated_height, rotated_width = img.shape[:2]
    img = cv2.resize(img, (int(rotated_width * scale), int(rotated_height * scale)))
    imgGray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    numberPlates = nPlateCascade.detectMultiScale(imgGray, 1.1, 10)
    for (x, y, w, h) in numberPlates:
        area = w * h
        if area > minArea:
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 255), 2)
            cv2.putText(img,
                        "Number Plate",
                        (x,y-5),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL,
                        1,
                        color,
                        2)
            imgRoi = img[y:y+h,x:x+w]
            # # Perspective correction
            # # find cornersc
            # pts1 = np.float32([[x, y], [x + w, y], [x, y + h], [x + w, y +h]])
            # width_plate = 200
            # height_plate = 50
            # pts2 = np.float32([[0,0], [width_plate, 0], [0, height_plate], [width_plate, height_plate]])
            # # calculate perspective transform matrix
            # persp_matrix = cv2.getPerspectiveTransform(pts1, pts2)
            # # apply transformation
            # imgWarp = cv2.warpPerspective(img, persp_matrix, (width_plate, height_plate))


            # Processing the detected number plate
            # imgDilate = cv2.cvtColor(imgWarp, cv2.COLOR_BGR2GRAY) # gray
            # imgDilate = cv2.adaptiveThreshold(imgDilate, 
            #                                   255, 
            #                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                   cv2.THRESH_BINARY,
            #                                   11,
            #                                   2) # base settings
            
            cv2.imshow("ROI", imgRoi)
            #cv2.imshow("ROI_threshold", imgDilate)

            # read the numbers
            results = reader.readtext(imgRoi)
            for (bbox, text, prob) in results:
                if prob > 0.5:
                    print(f"Распознанный номер: {text}")
                    number_dict[count] = text
                    count += 1

    cv2.imshow("Result", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break


cap.release()
cv2.destroyAllWindows()
