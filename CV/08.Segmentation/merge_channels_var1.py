import rasterio
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from utils import normalize

# santarosa
# paths to data
r_path = 'data/data_sources/santa_rosa/RED.tif'
g_path = 'data/data_sources/santa_rosa/GRN.tif'
b_path = 'data/data_sources/santa_rosa/BLUE.tif'

# read channels santarosa
with rasterio.open(r_path) as r:
    red = r.read(1)

with rasterio.open(g_path) as g:
    green = g.read(1)

with rasterio.open(b_path) as b:
    blue = b.read(1)

# merge to RGB
image = np.stack([red, green, blue], axis=-1) # H x W x 3

# visualise if need to check
# plt.imshow(image)
# plt.show()

# normalize
image_norm = normalize(image)

# save
path_to_save = 'data/data_sources/santa_rosa/santa_rosa_rgd.png'
Image.fromarray(image_norm).save(path_to_save)
print('santa_rosa_rgb saved')

# ventura
# path to data
r_path = 'data/data_sources/ventura/RED.tif'
g_path = 'data/data_sources/ventura/GRN.tif'
b_path = 'data/data_sources/ventura/BLUE.tif'

# read channels santarosa
with rasterio.open(r_path) as r:
    red = r.read(1)

with rasterio.open(g_path) as g:
    green = g.read(1)

with rasterio.open(b_path) as b:
    blue = b.read(1)

# merge to RGB
image = np.stack([red, green, blue], axis=-1) # H x W x 3

# visualise if need to check
# plt.imshow(image)
# plt.show()

# normalize
image_norm = normalize(image)

# save
path_to_save = 'data/data_sources/ventura/ventura_rgb.png'
Image.fromarray(image_norm).save(path_to_save)
print('ventura_rgb saved')