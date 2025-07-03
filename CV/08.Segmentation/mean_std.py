import os
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch
import numpy as np

# class
class SatelliteDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.png')]
        self.transform = transforms.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        return self.transform(image)

# parameters
root_dir = 'data/dataset/images'  # path to images
batch_size = 32

# create dataset and dataloader
dataset = SatelliteDataset(root_dir)
loader = DataLoader(dataset, batch_size=batch_size, num_workers=2)

# accum
mean = 0.
std = 0.
nb_samples = 0.

for data in loader:
    batch_samples = data.size(0)
    data = data.view(batch_samples, data.size(1), -1)  # [B, C, H*W]
    mean += data.mean(2).sum(0)
    std += data.std(2).sum(0)
    nb_samples += batch_samples

mean /= nb_samples
std /= nb_samples

print(f"Mean: {mean}")
print(f"Std: {std}")

"""
Mean: tensor([0.3527, 0.3395, 0.2912])
Std: tensor([0.1384, 0.1237, 0.1199])
"""