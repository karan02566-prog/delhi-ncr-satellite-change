"""BCE + Dice, both masking out INVALID_LABEL (255) pixels."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.data.label_generation import INVALID_LABEL


def masked_bce_dice_loss(logits: torch.Tensor, labels: torch.Tensor, dice_weight: float = 0.5):
    valid = labels != INVALID_LABEL
    targets = labels.clone().float()
    targets[~valid] = 0.0  # dummy value, excluded from loss via mask below

    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
    bce = (bce * valid.float()).sum() / valid.float().sum().clamp(min=1.0)

    probs = torch.sigmoid(logits)
    probs_v = probs[valid]
    targets_v = targets[valid]
    intersection = (probs_v * targets_v).sum()
    dice = 1 - (2 * intersection + 1e-6) / (probs_v.sum() + targets_v.sum() + 1e-6)

    return (1 - dice_weight) * bce + dice_weight * dice