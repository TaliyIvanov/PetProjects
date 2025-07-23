# imports
import os
from glob import glob

import matplotlib.pyplot as plt
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

# from project files
from src.datasets.datasets import SegmentationDataset
from src.metrics import compute_iou
from src.transforms.train_transforms import train_transform
from src.transforms.val_test_transforms import val_test_transform
from src.utils.utils import calculate_class_weights
from src.utils.utils import visualize_predictions

# configs
root_dir_images = "data/dataset/images"
root_dir_masks = "data/dataset/masks"

model_type = "Linknet"  # Unet, FPN, DeepLabV3Plus, UnetPlusPlus
encoder = "resnet34"
weights = "imagenet"
num_classes = 1
batch_size = 4
lr = 1e-3
EPOCHS = 100
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

save_path = f"best_model_{model_type.lower()}.pth"

# Model
ModelClass = getattr(smp, model_type)
model = ModelClass(
    encoder_name=encoder, encoder_weights=weights, in_channels=3, classes=num_classes
)

model.to(device)
# print(model)

# Data
image_paths = sorted(glob(os.path.join(root_dir_images, "*.png")))
mask_paths = sorted(glob(os.path.join(root_dir_masks, "*.png")))

# Train/Val/test split
# train (70%) и temp (30%)
train_imgs, temp_imgs, train_masks, temp_masks = train_test_split(
    image_paths, mask_paths, test_size=0.30, random_state=42
)

# from temp val (≈20%) и test (≈10%)
val_imgs, test_imgs, val_masks, test_masks = train_test_split(
    temp_imgs, temp_masks, test_size=1 / 3, random_state=42
)

# Datasets
train_dataset = SegmentationDataset(train_imgs, train_masks, transform=train_transform)
val_dataset = SegmentationDataset(val_imgs, val_masks, transform=val_test_transform)
test_dataset = SegmentationDataset(test_imgs, test_masks, transform=val_test_transform)

# Dataloaders
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

# pos weights
pos_weight = calculate_class_weights(train_masks, train_dataset, device=device)

# Loss & Metrics
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
metrics_fn = compute_iou
scheduler = ReduceLROnPlateau(
    optimizer, mode="max", factor=0.5, patience=2, verbose=True, min_lr=1e-6
)

# Lists for metrics
train_losses = []
val_losses = []
val_ious = []
lrs = []

# train
best_iou = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0

    pbar = tqdm(train_loader, desc=f"[Epoch {epoch + 1}/{EPOCHS}] Train")

    for images, masks in pbar:
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.unsqueeze(1).float()  # [B, 1, H, W]

        # forward
        preds = model(images)
        loss = loss_fn(preds, masks)

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        avg_train_loss = train_loss / len(train_loader)
        pbar.set_postfix({"loss": avg_train_loss})

    print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")

    # validation
    model.eval()
    val_loss = 0
    total_iou = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.unsqueeze(1).float()

            preds = model(images)
            loss = loss_fn(preds, masks)
            val_loss += loss.item()

            # Metrics
            preds_sigmoid = torch.sigmoid(preds)
            preds_binary = (preds_sigmoid > 0.5).float()
            iou = metrics_fn(preds_binary, masks)
            total_iou += iou.item()

    avg_val_loss = val_loss / len(val_loader)
    avg_iou = total_iou / len(val_loader)
    current_lr = optimizer.param_groups[0]["lr"]

    # ✅ Теперь сохраняем метрики
    train_losses.append(avg_train_loss)
    val_losses.append(avg_val_loss)
    val_ious.append(avg_iou)
    lrs.append(current_lr)

    print(f"[Epoch {epoch + 1}] Train Loss: {avg_train_loss:.4f}")
    print(f"[Epoch {epoch + 1}] Val Loss: {avg_val_loss:.4f} | IoU: {avg_iou:.4f}")

    scheduler.step(avg_iou)

    # Сохраняем лучшие веса
    if avg_iou > best_iou:
        best_iou = avg_iou
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path} (IoU improved to {avg_iou:.4f})")

# graphics

epochs_range = range(1, EPOCHS + 1)

plt.figure(figsize=(15, 5))

# Loss
plt.subplot(1, 3, 1)
plt.plot(epochs_range, train_losses, label="Train Loss")
plt.plot(epochs_range, val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.legend()

# IoU
plt.subplot(1, 3, 2)
plt.plot(epochs_range, val_ious, label="Val IoU", color="green")
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("Validation IoU")
plt.legend()

# LR
plt.subplot(1, 3, 3)
plt.plot(epochs_range, lrs, label="Learning Rate", color="orange")
plt.xlabel("Epoch")
plt.ylabel("LR")
plt.title("Learning Rate")
plt.legend()

plt.tight_layout()
plt.show()

# Evaluate on test set
model.eval()
test_iou = 0
test_loss = 0

with torch.no_grad():
    for images, masks in test_loader:
        images = images.to(device)
        masks = masks.to(device)
        masks = masks.unsqueeze(1).float()

        preds = model(images)
        loss = loss_fn(preds, masks)
        test_loss += loss.item()

        preds_sigmoid = torch.sigmoid(preds)
        preds_binary = (preds_sigmoid > 0.5).float()
        iou = metrics_fn(preds_binary, masks)
        test_iou += iou.item()

avg_test_loss = test_loss / len(test_loader)
avg_test_iou = test_iou / len(test_loader)

print(f"[Test] Loss: {avg_test_loss:.4f} | IoU: {avg_test_iou:.4f}")

# visualize model preds
visualize_predictions(model, test_loader, device, num_samples=5)
