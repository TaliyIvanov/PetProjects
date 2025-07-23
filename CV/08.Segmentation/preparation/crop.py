import cv2

# Load the image
path_to_image = "data/data_sources/ventura/all.tif"
img = cv2.imread(path_to_image)

if img is None:
    raise FileNotFoundError(f"Image not found at {path_to_image}")

img_w, img_h = img.shape[:2]

# Patch size
patch_size = 512
overlap = 0.5
step = int(patch_size * (1 - overlap))  # 256 for 50% overlap

# path to output
output_dit = "data/dataset/masks"  # check it

# counter for patch numbering
patch_id = 49  # write your number if you have more files

# Loop through the image with step size = patch size
for x in range(0, img_h - patch_size + 1, step):
    for y in range(0, img_w - patch_size + 1, step):
        # crop the patch
        patch = img[y : y + patch_size, x : x + patch_size]

        # save the patch
        patch_filename = f"{output_dit}/{patch_id}.png"
        cv2.imwrite(patch_filename, patch)

        # visualisation (draw a rectangle on the original image)
        # cv2.rectangle(img, (x, y), (x_end, y_end), (0,255,0), 2)

        patch_id += 1

# Show the original image with drawn patches
# cv2.imshow('Patches', img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
