import torch

def compute_iou(preds, masks, threshold=0.5):
    # Применяем sigmoid к логитам и применяем порог
    preds = torch.sigmoid(preds)  # Применяем сигмоид
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum((1, 2, 3))
    union = ((preds + masks) >= 1).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


__all__ = ['compute_iou']