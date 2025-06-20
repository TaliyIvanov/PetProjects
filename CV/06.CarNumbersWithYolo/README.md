Hello!
Here is the second part of my [project](https://github.com/TaliyIvanov/PetProjects/tree/main/CV/05.ObjectDetection/05.CarNumbersClassicMethods) to detect car numbers.
Before i use classic methods and EasyOCR. Now i will use pretraindes model Yolo 8 and EasyOCR to detect numbers.


I use the SORT from this [repo](https://github.com/abewley/sort/blob/master/sort.py), thanks.

A licensed plate detector was used to detect license plates. The model was trained with Yolov8. This the [repo](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8), where i find this model


### Problems
#### 1
AttributeError: module 'PIL.Image' has no attribute 'ANTIALIAS'

Two ways to fix:
##### 1
go to: 'venv/lib/easyocr/utils.py'
row 574: img = cv2.resize(img,(model_height,int(model_height*ratio)), interpolation=Image.ANTIALIAS)
change to: img = cv2.resize(img, (model_height, int(model_height*ratio)), interpolation=cv2.INTER_LANCZOS4)

row 576: img = cv2.resize(img,(int(model_height*ratio),model_height),interpolation=Image.ANTIALIAS)
change to: img = cv2.resize(img, (int(model_height*ratio), model_height), interpolation=cv2.INTER_LANCZOS4)

##### 2
pip uninstall pillow
pip install pillow==9.5.0