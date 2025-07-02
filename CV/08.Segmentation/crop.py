import cv2
import os

# Load the image
path_to_image = 'data/data_sources/ventura/all.tif'
img = cv2.imread(path_to_image)
img_w, img_h = img.shape[:2]

# Patch size
patch_w, patch_h = 512, 512

# path to output
output_dit = 'data/dataset/masks' # check it

# counter for patch numbering
patch_id = 64 # write your number if you have more files

# Loop through the image with step size = patch size
for x in range(0, patch_h*(img_h // patch_h), patch_h//2):
    for y in range(0, patch_w*(img_w // patch_w), patch_w//2):

        # Ensure patch does not exceed image boundaries
        x_end = min(x + patch_w//2, img_w)
        y_end = min(y + patch_h//2, img_h)

        # crop the patch
        patch = img[y:y_end, x:x_end]

        # save the patch
        patch_filename = f'{output_dit}/{patch_id}.png'
        cv2.imwrite(patch_filename, patch)

        # visualisation (draw a rectangle on the original image)
        # cv2.rectangle(img, (x, y), (x_end, y_end), (0,255,0), 2)

        patch_id += 1

# Show the original image with drawn patches
# cv2.imshow('Patches', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()