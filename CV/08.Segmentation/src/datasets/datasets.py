import cv2
import numpy as np
from torch.utils.data import Dataset


class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, transform=None):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        mask_path = self.mask_paths[idx]

        # Read image and mask
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.uint8)  # binary mask

        # Apply transforms
        if self.transform:
            # Pass both image and mask
            augmented = self.transform(image=image, mask=mask)
            # Extract results
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask


__all__ = ["SegmentationDataset"]
