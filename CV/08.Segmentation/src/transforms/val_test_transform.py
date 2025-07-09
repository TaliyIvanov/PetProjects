# imports
import albumentations as A
from albumentations.pytorch import ToTensorV2

"""
from mean_std.py
Mean: tensor([0.3527, 0.3395, 0.2912])
Std: tensor([0.1384, 0.1237, 0.1199])
"""

def val_test_transform(
        mean: list = (0.3527, 0.3395, 0.2912),
        std: list = (0.1384, 0.1237, 0.1199)
        ):
    
    result = A.Compose([
        A.Normalize(mean=mean, std=std),
        ToTensorV2()
        ])

    return result
    


# # transforms for validation
# val_test_transform = A.Compose([
#     A.Normalize(mean=(0.3527, 0.3395, 0.2912),
#                 std=(0.1384, 0.1237, 0.1199)),
#     ToTensorV2()
# ])

# __all__ = ['train_transform', 'val_transform']