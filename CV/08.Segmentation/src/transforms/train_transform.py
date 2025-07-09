# imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
from mean_std.py
Mean: tensor([0.3527, 0.3395, 0.2912])
Std: tensor([0.1384, 0.1237, 0.1199])
"""

def train_transform(
        horizontal_flip_p: float = 0.5,
        vertical_flip_p: float = 0.5,
        random_rotate: float = 0.5,
        shift_limit: float = 0.05,
        scale_limit: float = 0.1,
        rotate_limit: int = 15,
        shift_scale_rotate: float = 0.5,
        rand_bright_contrast: float = 0.2,
        grid_distortion: float = 0.2,
        mean: list = (0.3527, 0.3395, 0.2912),
        std: list = (0.1384, 0.1237, 0.1199)
        ):
        
        result = A.Compose([
            A.HorizontalFlip(p=horizontal_flip_p),
            A.VerticalFlip(p=vertical_flip_p),
            A.RandomRotate90(p=random_rotate),
            A.ShiftScaleRotate(shift_limit=shift_limit,
                            scale_limit=scale_limit,
                            rotate_limit=rotate_limit,
                            p=shift_scale_rotate),
            A.RandomBrightnessContrast(p=rand_bright_contrast),
            A.GridDistortion(p=grid_distortion),
            A.Normalize(mean=mean,
                        std=std),
            ToTensorV2()   
            ])
        
        return result




# # transforms for train
# train_transform = A.Compose([
#     A.HorizontalFlip(p=0.5),
#     A.VerticalFlip(p=0.5),
#     A.RandomRotate90(p=0.5),
#     A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=15, p=0.5),
#     A.RandomBrightnessContrast(p=0.2),
#     A.GridDistortion(p=0.2),
#     A.Normalize(mean=(0.3527, 0.3395, 0.2912),
#                 std=(0.1384, 0.1237, 0.1199)),
#     ToTensorV2()   
# ]) 