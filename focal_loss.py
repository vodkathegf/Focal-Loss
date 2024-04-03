# Focal Loss is not built-in in PyTorch so we are aware that this needs to be built.
# it is basically modified version of Binary Cross-Entropy Loss function

import torch
import torch.nn as nn
from torch import Tensor


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, reduction: str = "mean"):
        """
        Focal Loss
        designed to address imbalance in classes or hard-to-classify examples.

        params:
            -alpha: (float, optional), the class-spesific weighting factor to balance frequencies. Default: 0.25
            -gamma: (float, optional), the focusing parameter to adjust the rate where easy examples are down-weighted. Default: 2.0
            -reduction : it is the same parameter as in the other loss functions.

        """
        super(FocalLoss, self).__init__()

        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, y_true: Tensor, y_pred: Tensor) -> Tensor:
        """
        y_true: (Tensor), grount truth labels or targets
        y_pred: (Tensor), predicted logits or scores by the model
        """
        
        # let's calculate BCELoss at first
        bce_loss = nn.functional.binary_cross_entropy_with_logits(
            y_pred, y_true, reduction="none")

        pt = torch.exp(-bce_loss)
        modulating_factor = (1 - pt) ** self.gamma
        focal_loss = self.alpha * modulating_factor * bce_loss

        if self.reduction == "mean":
            focal_loss = torch.mean(focal_loss)

        elif self.reduction == "sum":
            focal_loss = torch.sum(focal_loss)

        return focal_loss
