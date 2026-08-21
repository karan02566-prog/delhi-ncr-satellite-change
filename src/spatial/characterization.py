"""Phase 8 — rule-based change characterization. Applied ONLY to
pixels already flagged as CHANGE by the model (Phase 7 predictions).
Descriptive layer, NOT a retrained multi-class model — per project
scope, multi-class labels aren't reliably achievable given Phase 3's
weakly-supervised label approach."""
import numpy as np

from src.data.label_generation import compute_ndvi, compute_ndbi

NO_CHANGE = 0
BUILTUP_EXPANSION = 1
VEGETATION_LOSS = 2
VEGETATION_GAIN = 3
OTHER_UNCERTAIN = 5


def characterize_change(t1_array, t2_array, change_mask: np.ndarray,
                         ndvi_thresh: float = 0.05, ndbi_thresh: float = 0.05):
    """
    change_mask: (H, W) bool — pixels predicted as CHANGE (from model,
    not ground-truth labels).
    Returns: (H, W) uint8 category map, values 0/1/2/3/5.
    Rule (only evaluated where change_mask is True):
      NDBI increased beyond threshold          -> built-up expansion
      NDVI decreased beyond threshold           -> vegetation loss
      NDVI increased beyond threshold           -> vegetation gain
      change flagged but neither index moved    -> other/uncertain
    Built-up expansion checked first: a pixel that both gains NDBI and
    loses NDVI (typical of construction over vegetation) is categorized
    as built-up expansion, the more specific/actionable label.
    """
    ndvi_t1, ndvi_t2 = compute_ndvi(t1_array), compute_ndvi(t2_array)
    ndbi_t1, ndbi_t2 = compute_ndbi(t1_array), compute_ndbi(t2_array)
    d_ndvi = ndvi_t2 - ndvi_t1
    d_ndbi = ndbi_t2 - ndbi_t1

    category = np.full(change_mask.shape, NO_CHANGE, dtype=np.uint8)

    builtup = change_mask & (d_ndbi > ndbi_thresh)
    veg_loss = change_mask & (~builtup) & (d_ndvi < -ndvi_thresh)
    veg_gain = change_mask & (~builtup) & (~veg_loss) & (d_ndvi > ndvi_thresh)
    other = change_mask & (~builtup) & (~veg_loss) & (~veg_gain)

    category[builtup] = BUILTUP_EXPANSION
    category[veg_loss] = VEGETATION_LOSS
    category[veg_gain] = VEGETATION_GAIN
    category[other] = OTHER_UNCERTAIN
    return category


def summarize_categories(category: np.ndarray, valid_mask: np.ndarray) -> dict:
    valid_total = valid_mask.sum()
    out = {}
    for name, val in [("builtup_expansion", BUILTUP_EXPANSION), ("vegetation_loss", VEGETATION_LOSS),
                       ("vegetation_gain", VEGETATION_GAIN), ("other_uncertain", OTHER_UNCERTAIN)]:
        count = int(((category == val) & valid_mask).sum())
        out[name] = {"pixel_count": count, "fraction_of_valid": count / valid_total if valid_total else 0.0}
    return out


def export_characterization_chips(t1_array, t2_array, category: np.ndarray, out_path: str):
    """Saves T1 | category map (color-coded) for visual sanity check."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    chip = t1_array[[2, 1, 0]]
    chip = np.transpose(chip, (1, 2, 0))
    chip = np.nan_to_num(chip, nan=0.0)
    p2, p98 = np.percentile(chip, [2, 98])
    t1_rgb = np.clip((chip - p2) / max(p98 - p2, 1e-6), 0, 1)

    colors = ["black", "red", "orange", "green", "black", "gray"]  # idx 0..5
    cmap = ListedColormap(colors)

    fig, axes = plt.subplots(1, 2, figsize=(7, 3.5))
    axes[0].imshow(t1_rgb); axes[0].set_title("T1"); axes[0].axis("off")
    axes[1].imshow(category, cmap=cmap, vmin=0, vmax=5)
    axes[1].set_title("Built-up(red) / Veg-loss(orange) / Veg-gain(green)")
    axes[1].axis("off")
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)