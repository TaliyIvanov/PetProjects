import segmentation_models_pytorch as smp
import torch.nn as nn


class Linknet(nn.Module):
    def __init__(self, encoder_name="resnet34", encoder_weights="imagenet", in_channels = 3, num_classes=1):
        super().__init__()
        self.in_channels = in_channels
        self.model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=self.in_channels,
            classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)
