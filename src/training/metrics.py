"""Precision/Recall/F1/IoU for the CHANGE class, masking invalid pixels."""
import torch
from src.data.label_generation import INVALID_LABEL


def compute_change_metrics(logits: torch.Tensor, labels: torch.Tensor, threshold: float = 0.5):
    valid = labels != INVALID_LABEL
    preds = (torch.sigmoid(logits) > threshold).long()
    targets = labels.clone()

    preds_v = preds[valid]
    targets_v = targets[valid]

    tp = ((preds_v == 1) & (targets_v == 1)).sum().item()
    fp = ((preds_v == 1) & (targets_v == 0)).sum().item()
    fn = ((preds_v == 0) & (targets_v == 1)).sum().item()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1, "iou": iou}