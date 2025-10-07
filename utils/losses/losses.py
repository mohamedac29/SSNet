import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['BCEDiceLoss', 'TverskyLoss', 'FocalDiceLoss']  # Add to exports


class FocalDiceLoss(nn.Module):
    def __init__(self, gamma=2, weight=0.7): # Increased weight on Dice component (1-weight)
        super().__init__()
        self.gamma = gamma  # Controls focus on hard examples
        self.weight = weight # Focal Loss component weight

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)

        # Flatten tensors
        inputs_flatten = inputs.view(-1)
        targets_flatten = targets.view(-1)

        # Focal Loss Component
        # Use BCEWithLogitsLoss for numerical stability
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs_flatten, targets_flatten, reduction='none'
        )
        pt = torch.exp(-bce_loss)  # Reduces loss for well-classified examples
        focal_loss = ((1 - pt) ** self.gamma * bce_loss).mean()

        # Dice Loss Component (calculated using sigmoid inputs)
        intersection = (inputs_sigmoid.view(-1) * targets_flatten).sum()
        dice_score = (2. * intersection + smooth) / (
                inputs_sigmoid.view(-1).sum() + targets_flatten.sum() + smooth
        )
        dice_loss = 1 - dice_score
        
        # Increased focus on the Dice/Tversky part for crack segmentation stability
        return self.weight * focal_loss + (1 - self.weight) * dice_loss


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, inputs, targets, smooth=1):
        inputs_sigmoid = torch.sigmoid(inputs)
        
        # BCE Loss
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='mean')
        
        # Dice Loss Component
        intersection = (inputs_sigmoid.view(-1) * targets.view(-1)).sum()
        dice_loss = 1 - (2. * intersection + smooth) / (
                inputs_sigmoid.view(-1).sum() + targets.view(-1).sum() + smooth
        )
        
        # Standard combined loss with equal weighting
        return bce_loss + dice_loss


def WeightedBCE(prediction, label, mask):
    """
    Original function definition preserved. Note: the `mask` input implies a custom
    weighting scheme which may be why standard losses are difficult.
    """
    num_positive = torch.sum(mask * label.float())
    num_negative = torch.sum(mask * (1. - label.float()))

    # Ensure prediction is treated as logit or probability consistent with use
    # Assuming prediction is logit for stability, but using torch.nn.functional.binary_cross_entropy
    # requires probabilities. If this loss is used, ensure prediction is sigmoid(model_output).
    # If prediction is model output (logit), use F.binary_cross_entropy_with_logits
    
    # Check if prediction is already sigmoided (0-1 range)
    # If not, this is numerically unstable. We assume it is sigmoided for the original code structure.
    cost = torch.nn.functional.binary_cross_entropy(
            prediction.float(),label.float(),weight = mask, reduce=False)
    
    return torch.sum(cost) /(num_positive+num_negative + 1e-6) # Added epsilon for stability


def BinaryFocalLoss(inputs, targets):
    # This function uses BCEWithLogitsLoss internally, expecting logits
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    BCE_loss = criterion(inputs, targets)
    pt = torch.exp(-BCE_loss)
    F_loss = (1-pt)**2 * BCE_loss
    return F_loss.mean()


def compute_kl_loss(p, q):
    p_loss = F.kl_div(F.log_softmax(p, dim=-1),
                      F.softmax(q, dim=-1), reduction='none')
    q_loss = F.kl_div(F.log_softmax(q, dim=-1),
                      F.softmax(p, dim=-1), reduction='none')

    p_loss = p_loss.mean()
    q_loss = q_loss.mean()

    loss = (p_loss + q_loss) / 2
    return loss


class TverskyLoss(nn.Module):
    # --- MAJOR FIX: Tuned for class imbalance (Cracks) ---
    def __init__(self, alpha=0.3, beta=0.7, smooth=1e-6): # Increased beta (FN penalty)
        super().__init__()
        self.alpha = alpha # Penalty for False Positives (background noise)
        self.beta = beta   # Penalty for False Negatives (missing the crack)
        self.smooth = smooth

    def forward(self, inputs, targets):
        inputs_sigmoid = torch.sigmoid(inputs) # Apply sigmoid internally

        # Flatten label and prediction tensors
        inputs_flatten = inputs_sigmoid.view(-1)
        targets_flatten = targets.view(-1)

        # True Positives, False Positives & False Negatives
        TP = (inputs_flatten * targets_flatten).sum()
        FP = ((1 - targets_flatten) * inputs_flatten).sum()
        FN = (targets_flatten * (1 - inputs_flatten)).sum()
       
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        
        return 1 - Tversky
