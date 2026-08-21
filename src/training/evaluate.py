"""Phase 6 — formal test-set evaluation: confusion matrix + FP/FN
spatial visualization for a trained model."""
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.data.label_generation import INVALID_LABEL


def compute_confusion_matrix(model, test_loader, device, siamese=False):
    tp = fp = fn = tn = 0
    with torch.no_grad():
        for batch in test_loader:
            if siamese:
                t1, t2, y = batch
                t1, t2, y = t1.to(device), t2.to(device), y.to(device)
                logits = model(t1, t2)
            else:
                x, y = batch
                x, y = x.to(device), y.to(device)
                logits = model(x)

            valid = y != INVALID_LABEL
            preds = (torch.sigmoid(logits) > 0.5).long()

            preds_v = preds[valid]
            targets_v = y[valid]

            tp += ((preds_v == 1) & (targets_v == 1)).sum().item()
            fp += ((preds_v == 1) & (targets_v == 0)).sum().item()
            fn += ((preds_v == 0) & (targets_v == 1)).sum().item()
            tn += ((preds_v == 0) & (targets_v == 0)).sum().item()

    return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


def print_confusion_matrix(cm: dict, model_name: str):
    total = sum(cm.values())
    print(f"\n{model_name} — Confusion Matrix (pixel-level, test set)")
    print(f"                Pred No-Change   Pred Change")
    print(f"Actual No-Change   {cm['tn']:>10}      {cm['fp']:>10}")
    print(f"Actual Change      {cm['fn']:>10}      {cm['tp']:>10}")
    print(f"Total valid pixels: {total}")
    print(f"False Positive rate: {cm['fp']/(cm['fp']+cm['tn']):.4f}")
    print(f"False Negative rate: {cm['fn']/(cm['fn']+cm['tp']):.4f}")


def export_fp_fn_chips(model, dataset, device, out_dir: str, n_samples: int = 8, siamese=False):
    """Saves T1 | T2 | GT label | prediction overlay (FP=blue, FN=orange)
    for a sample of test patches — visual error analysis."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    model.eval()

    idxs = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int)

    for i in idxs:
        if siamese:
            t1, t2, y = dataset[i]
            t1b, t2b = t1.unsqueeze(0).to(device), t2.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(t1b, t2b)
            t1_rgb_src = t1.numpy()
        else:
            x, y = dataset[i]
            xb = x.unsqueeze(0).to(device)
            with torch.no_grad():
                logits = model(xb)
            t1_rgb_src = x.numpy()[:6]  # first 6 channels are T1

        preds = (torch.sigmoid(logits) > 0.5).long().squeeze(0).cpu().numpy()
        label = y.numpy()
        valid = label != INVALID_LABEL

        def to_rgb(bands6):
            chip = bands6[[2, 1, 0]]  # B4,B3,B2 -> R,G,B
            chip = np.transpose(chip, (1, 2, 0))
            chip = np.nan_to_num(chip, nan=0.0)
            p2, p98 = np.percentile(chip, [2, 98])
            return np.clip((chip - p2) / max(p98 - p2, 1e-6), 0, 1)

        t1_rgb = to_rgb(t1_rgb_src)

        overlay = t1_rgb.copy()
        fp_mask = (preds == 1) & (label == 0) & valid
        fn_mask = (preds == 0) & (label == 1) & valid
        overlay[fp_mask] = [0.2, 0.4, 1.0]   # blue = false positive
        overlay[fn_mask] = [1.0, 0.55, 0.0]  # orange = false negative

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(t1_rgb); axes[0].set_title("T1"); axes[0].axis("off")
        axes[1].imshow(label, cmap="gray", vmin=0, vmax=1); axes[1].set_title("Ground truth"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title("FP (blue) / FN (orange)"); axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(out_path / f"error_chip_{i:04d}.png", dpi=120)
        plt.close(fig)

    print(f"Saved {len(idxs)} error-analysis chips to {out_dir}")