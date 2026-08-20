"""
Phase 3 — Ground truth / change labels for Delhi NCR Urban Change
Intelligence (2022 -> 2026).

Labels are SPECTRAL-INDEX-DERIVED, not manually painted. This is a
scientifically honest weakly-supervised approach, standard in remote
sensing when pixel-perfect ground truth isn't available:

  1. Compute delta-NDVI and delta-NDBI between T1 and T2.
  2. Threshold each delta at mean +/- k*std (k documented below),
     computed ONLY over valid (non-cloud-masked) pixels. This is a
     statistically adaptive threshold, not an arbitrary fixed number,
     and is recomputed and logged every run for reproducibility.
  3. A pixel is labeled "change" if it crosses EITHER threshold.
  4. A stratified sample of patches is exported as image chips for
     manual visual verification (see export_verification_chips).

Output label encoding (per-pixel, uint8):
    0   = no change
    1   = change
    255 = invalid / cloud-masked (excluded from loss during training)

Band order assumption (matches Phase 1/2 export): B2, B3, B4, B8, B11, B12
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional
import json
import numpy as np

DEFAULT_BAND_ORDER = ["B2", "B3", "B4", "B8", "B11", "B12"]
BAND_IDX = {name: i for i, name in enumerate(DEFAULT_BAND_ORDER)}

INVALID_LABEL = 255
NO_CHANGE_LABEL = 0
CHANGE_LABEL = 1


# ---------------------------------------------------------------------------
# Spectral indices
# ---------------------------------------------------------------------------

def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    return numerator / (denominator + eps)


def compute_ndvi(array: np.ndarray, band_order=DEFAULT_BAND_ORDER) -> np.ndarray:
    """array: (bands, H, W). Returns NDVI (H, W)."""
    idx = {n: i for i, n in enumerate(band_order)}
    nir, red = array[idx["B8"]], array[idx["B4"]]
    return _safe_ratio(nir - red, nir + red)


def compute_ndbi(array: np.ndarray, band_order=DEFAULT_BAND_ORDER) -> np.ndarray:
    """array: (bands, H, W). Returns NDBI (H, W)."""
    idx = {n: i for i, n in enumerate(band_order)}
    swir1, nir = array[idx["B11"]], array[idx["B8"]]
    return _safe_ratio(swir1 - nir, swir1 + nir)


def compute_valid_mask(t1_array: np.ndarray, t2_array: np.ndarray) -> np.ndarray:
    """True where ALL bands of BOTH T1 and T2 are finite (not cloud-masked/nodata)."""
    t1_valid = np.all(np.isfinite(t1_array), axis=0)
    t2_valid = np.all(np.isfinite(t2_array), axis=0)
    return t1_valid & t2_valid


# ---------------------------------------------------------------------------
# Change label generation
# ---------------------------------------------------------------------------

@dataclass
class ChangeLabelResult:
    label: np.ndarray            # (H, W) uint8, values {0, 1, 255}
    valid_mask: np.ndarray        # (H, W) bool
    delta_ndvi: np.ndarray        # (H, W) float
    delta_ndbi: np.ndarray        # (H, W) float
    thresholds: Dict[str, float] = field(default_factory=dict)  # logged for reproducibility


def compute_change_labels(
    t1_array: np.ndarray,
    t2_array: np.ndarray,
    ndvi_std_multiplier: float = 1.5,
    ndbi_std_multiplier: float = 1.5,
    band_order=DEFAULT_BAND_ORDER,
) -> ChangeLabelResult:
    """
    Threshold method: mean +/- k*std of the delta distribution over
    VALID pixels only. k=1.5 is a documented, commonly used starting
    point for unsupervised spectral change detection (roughly the
    inner ~87% of a normal distribution treated as "stable"; pixels
    beyond it flagged as change). This is intentionally conservative
    and will be corrected against the manual verification sample.

    A pixel is flagged CHANGE if it crosses EITHER the NDVI or NDBI
    threshold (vegetation change captures veg loss/gain; NDBI change
    captures built-up expansion). This is deliberately a BINARY core
    label. Change TYPE characterization (built-up vs veg vs other) is
    Phase 8, not Phase 3 -- do not conflate the two.
    """
    valid_mask = compute_valid_mask(t1_array, t2_array)

    ndvi_t1 = compute_ndvi(t1_array, band_order)
    ndvi_t2 = compute_ndvi(t2_array, band_order)
    ndbi_t1 = compute_ndbi(t1_array, band_order)
    ndbi_t2 = compute_ndbi(t2_array, band_order)

    delta_ndvi = ndvi_t2 - ndvi_t1
    delta_ndbi = ndbi_t2 - ndbi_t1

    ndvi_valid_vals = delta_ndvi[valid_mask]
    ndbi_valid_vals = delta_ndbi[valid_mask]

    ndvi_mean, ndvi_std = float(np.mean(ndvi_valid_vals)), float(np.std(ndvi_valid_vals))
    ndbi_mean, ndbi_std = float(np.mean(ndbi_valid_vals)), float(np.std(ndbi_valid_vals))

    ndvi_thr = ndvi_std_multiplier * ndvi_std
    ndbi_thr = ndbi_std_multiplier * ndbi_std

    ndvi_change = np.abs(delta_ndvi - ndvi_mean) > ndvi_thr
    ndbi_change = np.abs(delta_ndbi - ndbi_mean) > ndbi_thr

    label = np.full(delta_ndvi.shape, INVALID_LABEL, dtype=np.uint8)
    change_mask = (ndvi_change | ndbi_change) & valid_mask
    no_change_mask = (~(ndvi_change | ndbi_change)) & valid_mask
    label[change_mask] = CHANGE_LABEL
    label[no_change_mask] = NO_CHANGE_LABEL

    thresholds = {
        "ndvi_std_multiplier": ndvi_std_multiplier,
        "ndbi_std_multiplier": ndbi_std_multiplier,
        "ndvi_delta_mean": ndvi_mean,
        "ndvi_delta_std": ndvi_std,
        "ndvi_abs_threshold": ndvi_thr,
        "ndbi_delta_mean": ndbi_mean,
        "ndbi_delta_std": ndbi_std,
        "ndbi_abs_threshold": ndbi_thr,
        "valid_pixel_fraction": float(valid_mask.mean()),
        "change_pixel_fraction_of_valid": float(change_mask.sum() / max(valid_mask.sum(), 1)),
    }

    return ChangeLabelResult(
        label=label,
        valid_mask=valid_mask,
        delta_ndvi=delta_ndvi,
        delta_ndbi=delta_ndbi,
        thresholds=thresholds,
    )


# ---------------------------------------------------------------------------
# Patch-level labels (reuses preprocess.generate_patches for identical grid)
# ---------------------------------------------------------------------------

def label_patches(
    label_array: np.ndarray,
    patch_size: int = 256,
    stride: Optional[int] = None,
    min_valid_fraction: float = 0.5,
) -> List[Dict]:
    """
    Slices the label raster into the SAME row/col grid as
    preprocess.generate_patches() (must be called with identical
    patch_size/stride on the image arrays for row/col to line up).

    Returns list of dicts: row, col, valid_fraction, change_fraction,
    label_patch (uint8 array). Patches below min_valid_fraction
    (too much cloud-masked area) are dropped -- they'd contribute
    mostly ignored pixels to training and are not useful.
    """
    from src.data.preprocess import generate_patches

    label_3d = label_array[np.newaxis, :, :]  # (1, H, W) to match generate_patches signature
    raw_patches = generate_patches(label_3d, patch_size=patch_size, stride=stride)

    results = []
    for p in raw_patches:
        patch = p["patch"][0]  # back to (H, W)
        valid = patch != INVALID_LABEL
        valid_fraction = float(valid.mean())
        if valid_fraction < min_valid_fraction:
            continue
        change_fraction = float((patch[valid] == CHANGE_LABEL).mean()) if valid.any() else 0.0
        results.append({
            "row": p["row"],
            "col": p["col"],
            "valid_fraction": valid_fraction,
            "change_fraction": change_fraction,
            "label_patch": patch,
        })
    return results


# ---------------------------------------------------------------------------
# Manual verification sample (stratified by change fraction)
# ---------------------------------------------------------------------------

def select_verification_sample(
    patch_meta: List[Dict],
    n_per_stratum: int = 15,
    seed: int = 42,
) -> List[Dict]:
    """
    Stratifies patches into low/medium/high change-fraction terciles
    and randomly samples n_per_stratum from each, so the manual review
    set isn't dominated by trivially "no change" patches. Returns the
    sampled subset of patch_meta dicts (each gets a 'stratum' key added).
    """
    rng = np.random.default_rng(seed)
    change_fractions = np.array([p["change_fraction"] for p in patch_meta])
    if len(change_fractions) == 0:
        return []

    q1, q2 = np.quantile(change_fractions, [1 / 3, 2 / 3])
    strata = {"low": [], "medium": [], "high": []}
    for p in patch_meta:
        cf = p["change_fraction"]
        bucket = "low" if cf <= q1 else ("medium" if cf <= q2 else "high")
        strata[bucket].append(p)

    sample = []
    for name, items in strata.items():
        n = min(n_per_stratum, len(items))
        chosen_idx = rng.choice(len(items), size=n, replace=False)
        for i in chosen_idx:
            item = dict(items[i])
            item["stratum"] = name
            sample.append(item)
    return sample


def export_verification_chips(
    t1_array: np.ndarray,
    t2_array: np.ndarray,
    verification_sample: List[Dict],
    out_dir: str,
    patch_size: int,
    band_order=DEFAULT_BAND_ORDER,
) -> None:
    """
    For each sampled patch, saves a 3-panel PNG: T1 true-color RGB,
    T2 true-color RGB, label overlay (red = change) on T2. Also writes
    verification_log.csv with an empty 'reviewer_verdict' column for
    manual agree/disagree annotation.
    """
    import matplotlib.pyplot as plt
    import csv

    idx = {n: i for i, n in enumerate(band_order)}
    r, g, b = idx["B4"], idx["B3"], idx["B2"]

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def to_rgb(arr, row, col):
        chip = arr[[r, g, b], row:row + patch_size, col:col + patch_size]
        chip = np.transpose(chip, (1, 2, 0))
        chip = np.nan_to_num(chip, nan=0.0)
        p2, p98 = np.percentile(chip, [2, 98])
        chip = np.clip((chip - p2) / max(p98 - p2, 1e-6), 0, 1)
        return chip

    log_rows = []
    for i, item in enumerate(verification_sample):
        row, col = item["row"], item["col"]
        t1_rgb = to_rgb(t1_array, row, col)
        t2_rgb = to_rgb(t2_array, row, col)
        label_patch = item["label_patch"]

        overlay = t2_rgb.copy()
        change_pixels = label_patch == CHANGE_LABEL
        overlay[change_pixels] = [1.0, 0.0, 0.0]

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(t1_rgb); axes[0].set_title("T1 (2022)"); axes[0].axis("off")
        axes[1].imshow(t2_rgb); axes[1].set_title("T2 (2026)"); axes[1].axis("off")
        axes[2].imshow(overlay); axes[2].set_title(f"Change overlay ({item['stratum']})"); axes[2].axis("off")
        plt.tight_layout()

        fname = f"chip_{item['stratum']}_row{row}_col{col}.png"
        fig.savefig(out_path / fname, dpi=120)
        plt.close(fig)

        log_rows.append({
            "filename": fname, "row": row, "col": col,
            "stratum": item["stratum"], "change_fraction": round(item["change_fraction"], 4),
            "reviewer_verdict": "",  # fill in manually: agree / disagree / partial
        })

    with open(out_path / "verification_log.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
        writer.writeheader()
        writer.writerows(log_rows)


# ---------------------------------------------------------------------------
# Geographically separated train/val/test split
# ---------------------------------------------------------------------------

def geographic_split(
    patch_meta: List[Dict],
    raster_width: int,
    patch_size: int,
    train_frac: float = 0.70,
    val_frac: float = 0.15,
    test_frac: float = 0.15,
    buffer_px: Optional[int] = None,
) -> Dict[str, List[Dict]]:
    """
    Splits patches into train/val/test by COLUMN STRIPE (not random
    per-patch), so geographically adjacent patches can't end up in
    different splits. A buffer zone (default = 1 patch_size) is
    excluded entirely between stripes to prevent boundary leakage
    from overlapping context/spatial autocorrelation.

    Layout along width: [ TRAIN | buffer | VAL | buffer | TEST ]
    """
    assert abs(train_frac + val_frac + test_frac - 1.0) < 1e-6, "fractions must sum to 1"
    if buffer_px is None:
        buffer_px = patch_size

    train_end = raster_width * train_frac
    val_end = train_end + buffer_px + raster_width * val_frac

    train_zone = (0, train_end)
    val_zone = (train_end + buffer_px, val_end)
    test_zone = (val_end + buffer_px, raster_width)

    splits = {"train": [], "val": [], "test": [], "dropped_buffer": []}
    for p in patch_meta:
        col_start, col_end = p["col"], p["col"] + patch_size
        if train_zone[0] <= col_start and col_end <= train_zone[1]:
            splits["train"].append(p)
        elif val_zone[0] <= col_start and col_end <= val_zone[1]:
            splits["val"].append(p)
        elif test_zone[0] <= col_start and col_end <= test_zone[1]:
            splits["test"].append(p)
        else:
            splits["dropped_buffer"].append(p)

    return splits


def summarize_split(splits: Dict[str, List[Dict]]) -> Dict[str, Dict]:
    summary = {}
    for name, items in splits.items():
        if not items:
            summary[name] = {"count": 0}
            continue
        cf = [p["change_fraction"] for p in items]
        summary[name] = {
            "count": len(items),
            "mean_change_fraction": float(np.mean(cf)),
        }
    return summary


def save_split_manifest(splits: Dict[str, List[Dict]], thresholds: Dict, out_path: str) -> None:
    """Saves row/col/change_fraction per split + threshold metadata as JSON (not raw arrays -- keeps this file small)."""
    manifest = {"thresholds": thresholds, "splits": {}}
    for name, items in splits.items():
        manifest["splits"][name] = [
            {"row": p["row"], "col": p["col"], "change_fraction": p["change_fraction"], "valid_fraction": p["valid_fraction"]}
            for p in items
        ]
    with open(out_path, "w") as f:
        json.dump(manifest, f, indent=2)