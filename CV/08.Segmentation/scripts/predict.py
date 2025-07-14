import torch
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Сначала определю трансформации глобально
# в будущем необходимо поправить, чтобы они подтягивались из конфига
# val_transforms. На данный момент они идентичны

PREDICT_TRANSFORMS = A.Compose([
    A.Normalize(mean=[0.3527, 0.3395, 0.2912],
                std=[0.1384, 0.1237, 0.1199]),
    ToTensorV2()
])

device = "cuda" if torch.cuda.is_available else "cpu"

def load_model(model_path: str) -> torch.nn.Module:
    print(f"Loading model from {model_path} to {device}")
    model = torch.load(model_path, map_location=torch.device(device))
    model.to(device)
    model.eval()
    print("Model loaded succesfully!")
    return model

def predict_mask(model: torch.nn.Module, image: Image.Image) -> Image.Image:
    """
    Make predicts mask for one image.

    Args:
        model: Best trained model.
        image: Pil Image.

    Returns:
        Image with binary mask
    """

    # convert to numpy for transforms
    image_np = np.array(image.convert("RGB"))

    # transforms
    input_tensor = PREDICT_TRANSFORMS(image=image_np)["image"]
    input_tensor = input_tensor.unsqueeze(0).to(device) # add batch dimension
    
    with torch.no_grad():
        logits = model(input_tensor)

    preds = torch.sigmoid(logits).cpu()
    binary_mask = (preds > 0.5).squeeze().numpy().astype(np.uint8) * 255

    # convert numpy array to Pil Image
    mask_image = Image.fromarray(binary_mask, mode="L") # "L" for graysacale image

    return mask_image