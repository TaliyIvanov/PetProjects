import matplotlib.pyplot as plt
import torch

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

__all__ = ['visualize_predictions']

