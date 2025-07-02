import numpy as np
# auxiliary functions

# normalize to [0, 255]
def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= 255.0
    return img.astype(np.uint8)
