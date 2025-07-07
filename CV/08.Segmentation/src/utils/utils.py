# imports
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import albumentations as A
import cv2


# normalize to [0, 255]
def normalize(img):
    img = img.astype(np.float32)
    img -= img.min()
    img /= img.max()
    img *= 255.0
    return img.astype(np.uint8)

def calculate_class_weights(mask_paths, dataset, device="cpu", num_workers=4):
    """
    Рассчитывает вес для положительного класса (дома) на основе частоты классов.

    Args:
        mask_paths (list): Список путей к маскам.
        dataset (SegmentationDataset): Объект датасета.
        device (str): 'cuda' or 'cpu'.  Устройство, на котором будут вычисления.
        num_workers (int): Количество потоков для загрузки данных.

    Returns:
        torch.Tensor:  pos_weight
    """
    total_pixels = 0
    house_pixels = 0

    # Используем DataLoader для эффективной загрузки данных
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=num_workers)

    for _, masks in tqdm(dataloader, desc="Calculating class weights"):
        masks = masks.to(device)
        # Преобразуем в numpy array и подсчитываем пиксели
        masks_np = masks.cpu().numpy()  # Переносим на CPU, если GPU нет
        for mask_np in masks_np:
            total_pixels += mask_np.size # Общее количество пикселей в маске
            house_pixels += np.sum(mask_np) # Количество пикселей дома (класс 1)

    non_house_pixels = total_pixels - house_pixels
    if house_pixels == 0:
         return torch.tensor([1.0]).to(device) # если домов нет, ставим вес 1.0.  Это может быть артефакт данных.
    pos_weight = torch.tensor([non_house_pixels / house_pixels]).to(device)
    return pos_weight


# Визуализация предсказаний модели на тестовой выборке
def visualize_predictions(model, dataloader, device, num_samples=5):
    model.eval()
    plt.figure(figsize=(12, num_samples * 3))

    count = 0
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            masks = masks.unsqueeze(1).float()

            preds = model(images)
            preds = torch.sigmoid(preds)
            preds = (preds > 0.5).float()

            for i in range(images.size(0)):
                if count >= num_samples:
                    break

                img = images[i].cpu().permute(1, 2, 0).numpy()
                true_mask = masks[i][0].cpu().numpy()
                pred_mask = preds[i][0].cpu().numpy()

                # Показать 3 изображения: оригинал, ground truth, prediction
                plt.subplot(num_samples, 3, count * 3 + 1)
                plt.imshow(img)
                plt.title('Image')
                plt.axis('off')

                plt.subplot(num_samples, 3, count * 3 + 2)
                plt.imshow(true_mask, cmap='gray')
                plt.title('Ground Truth')
                plt.axis('off')

                plt.subplot(num_samples, 3, count * 3 + 3)
                plt.imshow(pred_mask, cmap='gray')
                plt.title('Prediction')
                plt.axis('off')

                count += 1

            if count >= num_samples:
                break

    plt.tight_layout()
    plt.show()

# for visualize aigmentatuons data

# Simple function to overlay mask on image for visualization
def overlay_mask(image, mask, alpha=0.5, color=(0, 1, 0)): # Green overlay
    # Convert mask to 3 channels if needed, ensure boolean type
    mask_overlay = np.zeros_like(image, dtype=np.uint8)
    # Create a color overlay where mask is > 0
    mask_overlay[mask > 0] = (np.array(color) * 255).astype(np.uint8)

    # Blend image and overlay
    overlayed_image = cv2.addWeighted(image, 1, mask_overlay, alpha, 0)
    return overlayed_image


def visualize_segmentation(dataset, idx=0, samples=3):
    # Make a copy of the transform list to modify for visualization
    if isinstance(dataset.transform, A.Compose):
        vis_transform_list = [
            t for t in dataset.transform
            if not isinstance(t, (A.Normalize, A.ToTensorV2))
        ]
        vis_transform = A.Compose(vis_transform_list)
    else:
        print("Warning: Could not automatically strip Normalize/ToTensor for visualization.")
        vis_transform = dataset.transform

    figure, ax = plt.subplots(samples + 1, 2, figsize=(8, 4 * (samples + 1)))

    # --- Get the original image and mask --- #
    original_transform = dataset.transform
    dataset.transform = None # Temporarily disable for raw data access
    image, mask = dataset[idx]
    dataset.transform = original_transform # Restore

    # Display original
    ax[0, 0].imshow(image)
    ax[0, 0].set_title("Original Image")
    ax[0, 0].axis("off")
    ax[0, 1].imshow(mask, cmap='gray') # Show mask directly
    ax[0, 1].set_title("Original Mask")
    ax[0, 1].axis("off")
    # ax[0, 1].imshow(overlay_mask(image, mask)) # Or show overlay
    # ax[0, 1].set_title("Original Overlay")

    # --- Apply and display augmented versions --- #
    for i in range(samples):
        # Apply the visualization transform
        if vis_transform:
            augmented = vis_transform(image=image, mask=mask)
            aug_image = augmented['image']
            aug_mask = augmented['mask']
        else:
            aug_image, aug_mask = image, mask # Should not happen normally

        # Display augmented image and mask
        ax[i + 1, 0].imshow(aug_image)
        ax[i + 1, 0].set_title(f"Augmented Image {i+1}")
        ax[i + 1, 0].axis("off")

        ax[i + 1, 1].imshow(aug_mask, cmap='gray') # Show mask directly
        ax[i + 1, 1].set_title(f"Augmented Mask {i+1}")
        ax[i + 1, 1].axis("off")
        # ax[i+1, 1].imshow(overlay_mask(aug_image, aug_mask)) # Or show overlay
        # ax[i+1, 1].set_title(f"Augmented Overlay {i+1}")


    plt.tight_layout()
    plt.show()

# Assuming train_dataset is created with train_transform:
# visualize_segmentation(train_dataset, samples=3)


__all__ = ['normalize', 
           'calculate_class_weights', 
           'visualize_predictions', 
           'visualize_segmentation']