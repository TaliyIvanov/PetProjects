# imports
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import numpy as np

"""
from mean_std.py
Mean: tensor([0.3527, 0.3395, 0.2912])
Std: tensor([0.1384, 0.1237, 0.1199])
"""

# transforms for train
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),
    A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.GridDistortion(p=0.2),
    A.Normalize(mean=(0.3527, 0.3395, 0.2912),
                std=(0.1384, 0.1237, 0.1199)),
    ToTensorV2()   
])

# transforms for validation
val_transform = A.Compose([
    A.Normalize(mean=(0.3527, 0.3395, 0.2912),
                std=(0.1384, 0.1237, 0.1199)),
    ToTensorV2()
])

__all__ = ['train_transform', 'val_transform']