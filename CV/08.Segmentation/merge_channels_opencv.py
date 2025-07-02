import cv2
import numpy as np
from utils import normalize

# santarosa
# paths to data
r_path = 'data/data_sources/santa_rosa/RED.tif'
g_path = 'data/data_sources/santa_rosa/GRN.tif'
b_path = 'data/data_sources/santa_rosa/BLUE.tif'
output_path = 'data/data_sources/santa_rosa/santa_rosa_rgb_opencv.png'

# read channels 
red = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
green = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
blue = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

# merge to RGB
image = cv2.merge([blue, green, red])

# visualise if need to check
# cv2.imshow("Image", image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# normalize
image_norm = normalize(image)

# save
cv2.imwrite(output_path, image_norm)
print('santa_rosa_rgb_opencv saved')

# -----------------------------------------------------------------

# ventura
# paths to data
r_path = 'data/data_sources/ventura/RED.tif'
g_path = 'data/data_sources/ventura/GRN.tif'
b_path = 'data/data_sources/ventura/BLUE.tif'
output_path = 'data/data_sources/ventura/ventura_rgb_opencv.png'

# read channels 
red = cv2.imread(r_path, cv2.IMREAD_GRAYSCALE)
green = cv2.imread(g_path, cv2.IMREAD_GRAYSCALE)
blue = cv2.imread(b_path, cv2.IMREAD_GRAYSCALE)

# merge to RGB
image = cv2.merge([blue, green, red])

# visualise if need to check
# cv2.imshow("Image", image) 
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# normalize
image_norm = normalize(image)

# save
cv2.imwrite(output_path, image_norm)
print('ventura_rgb_opencv saved')