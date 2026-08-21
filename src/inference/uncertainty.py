"""Phase 7 — MC Dropout uncertainty. Runs N stochastic forward passes
with dropout ACTIVE at inference (model.train(), not model.eval()),
computes mean probability map + std deviation (uncertainty map).

IMPORTANT: this is MODEL uncertainty (how much the network's own
stochastic dropout sampling disagrees with itself), NOT a calibrated
estimate of real-world/ground-truth uncertainty. Document this
distinction wherever uncertainty maps are shown."""
import torch
import numpy as np


def mc_dropout_predict(model, x: torch.Tensor, n_passes: int = 20):
    """
    x: (B, C, H, W) input tensor, already on the correct device.
    Returns: mean_prob (B, H, W), std_prob (B, H, W) — both numpy arrays.
    """
    model.train()  # keeps Dropout2d ACTIVE — this is the whole point of MC Dropout
    # BatchNorm layers also switch to train-mode batch stats under
    # model.train(); with batch_size>=2 this is a known, accepted
    # approximation for MC Dropout and does not invalidate the method.

    probs = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits = model(x)
            probs.append(torch.sigmoid(logits).cpu().numpy())

    probs = np.stack(probs, axis=0)  # (n_passes, B, H, W)
    mean_prob = probs.mean(axis=0)
    std_prob = probs.std(axis=0)
    return mean_prob, std_prob


def export_uncertainty_chips(model, dataset, device, out_dir: str,
                              n_samples: int = 8, n_passes: int = 20):
    """Saves T1 | mean prediction | uncertainty (std) heatmap for a
    sample of patches."""
    import matplotlib.pyplot as plt
    from pathlib import Path

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    idxs = np.linspace(0, len(dataset) - 1, min(n_samples, len(dataset)), dtype=int)

    for i in idxs:
        x, y = dataset[i]
        xb = x.unsqueeze(0).to(device)
        mean_prob, std_prob = mc_dropout_predict(model, xb, n_passes=n_passes)
        mean_prob, std_prob = mean_prob[0], std_prob[0]

        t1_rgb_src = x.numpy()[:6]
        chip = t1_rgb_src[[2, 1, 0]]
        chip = np.transpose(chip, (1, 2, 0))
        chip = np.nan_to_num(chip, nan=0.0)
        p2, p98 = np.percentile(chip, [2, 98])
        t1_rgb = np.clip((chip - p2) / max(p98 - p2, 1e-6), 0, 1)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(t1_rgb); axes[0].set_title("T1"); axes[0].axis("off")
        axes[1].imshow(mean_prob, cmap="viridis", vmin=0, vmax=1); axes[1].set_title("Mean change prob"); axes[1].axis("off")
        im = axes[2].imshow(std_prob, cmap="magma"); axes[2].set_title("Uncertainty (std)"); axes[2].axis("off")
        plt.tight_layout()
        fig.savefig(out_path / f"uncertainty_chip_{i:04d}.png", dpi=120)
        plt.close(fig)

    print(f"Saved {len(idxs)} uncertainty chips to {out_dir}")