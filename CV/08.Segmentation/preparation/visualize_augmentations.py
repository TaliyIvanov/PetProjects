# to check augmentations

import os
from glob import glob
from src.transforms.val_test_transform import train_transform
from src.datasets.datasets import SegmentationDataset
from src.utils.utils import visualize_segmentation

# Путь к данным
image_dir = 'data/dataset/images'
mask_dir = 'data/dataset/masks'

# Получаем пути к изображениям и маскам
image_paths = sorted(glob(os.path.join(image_dir, '*.png')))  # или .jpg, если нужно
mask_paths = sorted(glob(os.path.join(mask_dir, '*.png')))    # должно совпадать по имени с изображениями

assert len(image_paths) == len(mask_paths), "Количество изображений и масок не совпадает"

# Создаем датасет с трансформациями
dataset = SegmentationDataset(image_paths=image_paths, mask_paths=mask_paths, transform=train_transform)

# Визуализируем 3 примера
visualize_segmentation(dataset, idx=0, samples=3)