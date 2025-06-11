
## This code from gfg
## Link: https://www.geeksforgeeks.org/python-opencv-capture-video-from-camera/
# import cv2 as cv

# # Open the default camera
# cam = cv.VideoCapture(0)

# # Get the default frame width and height
# frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
# frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))

# # Define the codec and create VideoWriter object

# fourcc = cv.VideoWriter_fourcc(*'mp4v')
# out = cv.VideoWriter('output.mp4', fourcc, 30.0, (frame_width, frame_height))

# while True:
#     ret, frame = cam.read()
#     # Write the frame to the output filr
#     # out.write(frame)

#     # Display the captured frame
#     cv.imshow('Camera', frame)

#     # Press 'q' to exit the loop
#     if cv.waitKey(1) == ord('q'):
#         break


# # Release the capture and writer objects
# cam.release()
# # out.release()
# cv.destroyAllWindows()


# This code from official docs
# Link: https://docs.opencv.org/4.x/dd/d43/tutorial_py_video_display.html
import numpy as np
import cv2 as cv

cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Cannot open camera")
    exit()
while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    # Our operations on the frame come here
    # gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # Display the resulting frame
    cv.imshow('frame', frame)
    if cv.waitKey(1) == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv.destroyAllWindows()