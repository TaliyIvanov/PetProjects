import torch

def compute_iou(preds: torch.Tensor, masks: torch.Tensor, threshold: float=0.5) -> float:
    # Применяем sigmoid к логитам и применяем порог
    preds = torch.sigmoid(preds)  # Применяем сигмоид
    preds = (preds > threshold).float()
    intersection = (preds * masks).sum((1, 2, 3))
    union = ((preds + masks) >= 1).float().sum((1, 2, 3))
    iou = (intersection + 1e-6) / (union + 1e-6)
    return iou.mean()


__all__ = ['compute_iou']