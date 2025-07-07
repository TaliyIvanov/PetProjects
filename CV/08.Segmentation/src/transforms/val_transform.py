# imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
from mean_std.py
Mean: tensor([0.3527, 0.3395, 0.2912])
Std: tensor([0.1384, 0.1237, 0.1199])
"""

# transforms for validation
val_transform = A.Compose([
    A.Normalize(mean=(0.3527, 0.3395, 0.2912),
                std=(0.1384, 0.1237, 0.1199)),
    ToTensorV2()
])

# __all__ = ['train_transform', 'val_transform']