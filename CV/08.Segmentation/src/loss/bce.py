import torch.nn as nn

class BCEWithLogitsLoss(nn.module):
    def __init__(self):
        super().__init__()
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, preds, targets):
        return self.loss_fn(preds,targets)