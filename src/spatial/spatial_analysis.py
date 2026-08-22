"""
Phase 9 — Full-Scene Spatial Intelligence Pipeline & Aggregations.

Orchestrates full-scene inference and spatial analytics across all usable
patches (~1008 patches covering the entire Delhi NCR study ROI):
  - Streaming MC Dropout inference (N=20 stochastic forward passes, model.train()).
  - Rule-based change characterization per patch.
  - Crash-resilient incremental result saving (JSONL/CSV/NPZ) with resume support.
  - Full-scene aggregate physical area calculations (km², ha, %).
  - Dedicated test-split sanity check (n=56) against Phase 3/6/7 locked metrics.
  - Full-scene and test-set spatial visualizations and export artifacts.
"""

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import torch
from torch.utils.data import Dataset, DataLoader

from src.data.label_generation import compute_valid_mask, INVALID_LABEL
from src.inference.uncertainty import mc_dropout_predict
from src.spatial.characterization import (
    characterize_change,
    BUILTUP_EXPANSION,
    VEGETATION_LOSS,
    VEGETATION_GAIN,
    OTHER_UNCERTAIN,
    NO_CHANGE,
)
from src.spatial.hotspots import (
    calculate_pixel_area,
    compute_patch_geometries,
    detect_spatial_hotspots,
    derive_adaptive_urban_cutoff,
    compute_urban_edge_proximity,
    export_urban_cutoff_verification_chips,
)


class FullScenePatchDataset(Dataset):
    """
    Dataset iterating across all usable patches in the scene (~1008 patches).
    Can slice dynamically from in-memory (t1_array, t2_array) or load from cache.
    """
    def __init__(self,
                 patch_records: List[Dict],
                 t1_array: Optional[np.ndarray] = None,
                 t2_array: Optional[np.ndarray] = None,
                 cache_dir: Optional[str] = None,
                 patch_size: int = 256):
        self.records = patch_records
        self.t1_array = t1_array
        self.t2_array = t2_array
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.patch_size = patch_size

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        item = self.records[idx]
        r, c = item["row"], item["col"]

        if self.t1_array is not None and self.t2_array is not None:
            t1p = self.t1_array[:, r:r + self.patch_size, c:c + self.patch_size]
            t2p = self.t2_array[:, r:r + self.patch_size, c:c + self.patch_size]
        elif self.cache_dir is not None:
            split = item.get("split", "all")
            fname = f"patch_{idx:05d}_r{r}_c{c}.npz"
            fpath = self.cache_dir / split / fname
            if not fpath.exists():
                matches = list(self.cache_dir.glob(f"*/patch_*_r{r}_c{c}.npz"))
                if matches:
                    fpath = matches[0]
                else:
                    raise FileNotFoundError(f"Patch file not found: {fname} in {self.cache_dir}")
            data = np.load(fpath)
            t1p, t2p = data["t1"], data["t2"]
        else:
            raise ValueError("Either (t1_array, t2_array) or cache_dir must be provided.")

        x = np.concatenate([t1p, t2p], axis=0).astype(np.float32)  # (12, H, W)
        x = np.nan_to_num(x, nan=0.0)

        meta = {
            "patch_index": idx,
            "row": r,
            "col": c,
            "split": item.get("split", "unknown"),
        }
        return torch.from_numpy(x), t1p, t2p, meta


def run_spatial_pipeline(
    model: torch.nn.Module,
    patch_records: List[Dict],
    t1_array: np.ndarray,
    t2_array: np.ndarray,
    device: torch.device,
    out_dir: str,
    n_passes: int = 20,
    batch_size: int = 4,
    resume: bool = True,
    transform: Optional[object] = None,
    ndvi_thresh: float = 0.05,
    ndbi_thresh: float = 0.05,
    patch_size: int = 256,
) -> Tuple[pd.DataFrame, Dict, Dict, np.ndarray]:
    """
    Executes full-scene MC Dropout inference and rule-based change characterization
    across all usable patches (~1008), with crash-resilient incremental saving.

    Args:
        model: Trained BaselineChangeCNN (with dropout).
        patch_records: List of dicts with 'row', 'col', and optional 'split'.
        t1_array, t2_array: (6, H, W) full scene rasters.
        device: torch.device ('cuda' or 'cpu').
        out_dir: Directory to save streaming and final outputs.
        n_passes: MC Dropout stochastic forward passes (default 20).
        batch_size: Batch size for forward passes.
        resume: If True, skips patches already recorded in streaming log.
        transform: rasterio affine transform.
        ndvi_thresh, ndbi_thresh: Thresholds for change characterization (default 0.05).
        patch_size: Square patch side length (default 256).

    Returns:
        patch_df: DataFrame of all processed patch metrics.
        full_scene_summary: Aggregate physical area metrics across all usable patches.
        test_sanity_summary: Sanity check metrics for the test split.
        full_builtup_mask: (H, W) boolean mask of model-predicted built-up expansion pixels.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    streaming_file = out_path / "patch_results_streaming.jsonl"
    masks_file = out_path / "full_scene_masks.npz"

    H, W = t1_array.shape[1], t1_array.shape[2]
    full_builtup_mask = np.zeros((H, W), dtype=bool)

    # Check for existing completed patch indices to support resume
    processed_indices = set()
    existing_records = []
    if resume and streaming_file.exists():
        with open(streaming_file, "r") as f:
            for line in f:
                if line.strip():
                    try:
                        rec = json.loads(line)
                        processed_indices.add(rec["patch_index"])
                        existing_records.append(rec)
                    except json.JSONDecodeError:
                        pass
        print(f"Resuming: found {len(processed_indices)} already processed patches.")
        if masks_file.exists():
            try:
                mdata = np.load(masks_file)
                if "builtup_mask" in mdata:
                    full_builtup_mask = mdata["builtup_mask"].copy()
            except Exception:
                pass

    dataset = FullScenePatchDataset(
        patch_records=patch_records,
        t1_array=t1_array,
        t2_array=t2_array,
        patch_size=patch_size,
    )

    model.train()  # Required for MC Dropout active sampling
    model.to(device)

    streaming_fp = open(streaming_file, "a" if resume else "w", buffering=1)
    new_records = []

    total_patches = len(dataset)
    start_time = time.time()

    for idx in range(total_patches):
        if idx in processed_indices:
            continue

        x_tensor, t1p, t2p, meta = dataset[idx]
        x_batch = x_tensor.unsqueeze(0).to(device)  # (1, 12, H, W)

        # MC Dropout forward passes
        mean_prob, std_prob = mc_dropout_predict(model, x_batch, n_passes=n_passes)
        mean_prob = mean_prob[0]  # (H, W)
        std_prob = std_prob[0]   # (H, W)

        # Valid mask over patch
        valid_mask = compute_valid_mask(t1p, t2p)
        valid_pixels = int(valid_mask.sum())

        # Predicted change mask (threshold 0.5 on mean probability)
        pred_change = (mean_prob >= 0.5) & valid_mask
        change_pixels = int(pred_change.sum())

        # Rule-based characterization
        category_map = characterize_change(
            t1p, t2p, pred_change,
            ndvi_thresh=ndvi_thresh,
            ndbi_thresh=ndbi_thresh,
        )

        builtup_mask_patch = (category_map == BUILTUP_EXPANSION) & valid_mask
        builtup_px = int(builtup_mask_patch.sum())
        veg_loss_px = int(((category_map == VEGETATION_LOSS) & valid_mask).sum())
        veg_gain_px = int(((category_map == VEGETATION_GAIN) & valid_mask).sum())
        other_px = int(((category_map == OTHER_UNCERTAIN) & valid_mask).sum())
        no_change_px = int(((category_map == NO_CHANGE) & valid_mask).sum())

        # Update full-scene built-up expansion mask
        r, c = meta["row"], meta["col"]
        full_builtup_mask[r:r + patch_size, c:c + patch_size] |= builtup_mask_patch

        # MC Dropout uncertainty breakdown
        unc_change = float(std_prob[pred_change].mean()) if change_pixels > 0 else 0.0
        unc_nochange = float(std_prob[~pred_change & valid_mask].mean()) if (valid_pixels - change_pixels) > 0 else 0.0
        unc_all = float(std_prob[valid_mask].mean()) if valid_pixels > 0 else 0.0

        if transform is not None:
            cx, cy = transform * (c + patch_size / 2.0, r + patch_size / 2.0)
        else:
            cx, cy = c + patch_size / 2.0, r + patch_size / 2.0

        rec = {
            "patch_index": idx,
            "row": r,
            "col": c,
            "split": meta["split"],
            "valid_pixels": valid_pixels,
            "change_pixels": change_pixels,
            "no_change_pixels": no_change_px,
            "builtup_pixels": builtup_px,
            "veg_loss_pixels": veg_loss_px,
            "veg_gain_pixels": veg_gain_px,
            "other_pixels": other_px,
            "change_fraction": float(change_pixels / max(valid_pixels, 1)),
            "builtup_fraction": float(builtup_px / max(valid_pixels, 1)),
            "mean_uncertainty_all": unc_all,
            "mean_uncertainty_change": unc_change,
            "mean_uncertainty_nochange": unc_nochange,
            "centroid_x": cx,
            "centroid_y": cy,
        }

        # Write streaming JSON line
        streaming_fp.write(json.dumps(rec) + "\n")
        new_records.append(rec)

        if (idx + 1) % 50 == 0 or (idx + 1) == total_patches:
            elapsed = time.time() - start_time
            print(f"Progress: [{idx + 1}/{total_patches}] patches processed ({elapsed:.1f}s)")

    streaming_fp.close()

    # Save full built-up mask array
    np.savez_compressed(masks_file, builtup_mask=full_builtup_mask)

    all_records = existing_records + new_records
    all_records.sort(key=lambda x: x["patch_index"])
    patch_df = pd.DataFrame(all_records)

    # Save full CSV
    patch_df.to_csv(out_path / "patch_spatial_metrics.csv", index=False)

    # -----------------------------------------------------------------------
    # Aggregate Full-Scene Statistics
    # -----------------------------------------------------------------------
    tot_valid = int(patch_df["valid_pixels"].sum())
    tot_change = int(patch_df["change_pixels"].sum())
    tot_builtup = int(patch_df["builtup_pixels"].sum())
    tot_veg_loss = int(patch_df["veg_loss_pixels"].sum())
    tot_veg_gain = int(patch_df["veg_gain_pixels"].sum())
    tot_other = int(patch_df["other_pixels"].sum())
    tot_no_change = int(patch_df["no_change_pixels"].sum())

    area_valid = calculate_pixel_area(tot_valid)
    area_change = calculate_pixel_area(tot_change)
    area_builtup = calculate_pixel_area(tot_builtup)
    area_veg_loss = calculate_pixel_area(tot_veg_loss)
    area_veg_gain = calculate_pixel_area(tot_veg_gain)
    area_other = calculate_pixel_area(tot_other)

    full_scene_summary = {
        "scope": "full_scene_all_usable_patches",
        "total_patches": len(patch_df),
        "total_valid_km2": float(area_valid["area_km2"]),
        "total_valid_ha": float(area_valid["area_ha"]),
        "total_changed_km2": float(area_change["area_km2"]),
        "total_changed_ha": float(area_change["area_ha"]),
        "overall_change_percentage": float(tot_change / max(tot_valid, 1) * 100.0),
        "categories": {
            "builtup_expansion": {
                "pixels": tot_builtup,
                "area_km2": float(area_builtup["area_km2"]),
                "area_ha": float(area_builtup["area_ha"]),
                "pct_of_valid_area": float(tot_builtup / max(tot_valid, 1) * 100.0),
                "pct_of_total_change": float(tot_builtup / max(tot_change, 1) * 100.0),
            },
            "vegetation_loss": {
                "pixels": tot_veg_loss,
                "area_km2": float(area_veg_loss["area_km2"]),
                "area_ha": float(area_veg_loss["area_ha"]),
                "pct_of_valid_area": float(tot_veg_loss / max(tot_valid, 1) * 100.0),
                "pct_of_total_change": float(tot_veg_loss / max(tot_change, 1) * 100.0),
            },
            "vegetation_gain": {
                "pixels": tot_veg_gain,
                "area_km2": float(area_veg_gain["area_km2"]),
                "area_ha": float(area_veg_gain["area_ha"]),
                "pct_of_valid_area": float(tot_veg_gain / max(tot_valid, 1) * 100.0),
                "pct_of_total_change": float(tot_veg_gain / max(tot_change, 1) * 100.0),
            },
            "other_uncertain": {
                "pixels": tot_other,
                "area_km2": float(area_other["area_km2"]),
                "area_ha": float(area_other["area_ha"]),
                "pct_of_valid_area": float(tot_other / max(tot_valid, 1) * 100.0),
                "pct_of_total_change": float(tot_other / max(tot_change, 1) * 100.0),
            },
        },
        "mean_uncertainty_change": float(patch_df["mean_uncertainty_change"].mean()),
        "mean_uncertainty_nochange": float(patch_df["mean_uncertainty_nochange"].mean()),
    }

    # -----------------------------------------------------------------------
    # Test Split Sanity Check (n=56)
    # -----------------------------------------------------------------------
    test_df = patch_df[patch_df["split"] == "test"]
    t_valid = int(test_df["valid_pixels"].sum())
    t_change = int(test_df["change_pixels"].sum())
    t_builtup = int(test_df["builtup_pixels"].sum())

    test_sanity_summary = {
        "scope": "test_split_sanity_check",
        "patch_count": len(test_df),
        "total_valid_pixels": t_valid,
        "total_change_pixels": t_change,
        "test_change_rate": float(t_change / max(t_valid, 1)),
        "test_builtup_rate": float(t_builtup / max(t_valid, 1)),
        "mean_uncertainty_change": float(test_df["mean_uncertainty_change"].mean()) if len(test_df) else 0.0,
        "mean_uncertainty_nochange": float(test_df["mean_uncertainty_nochange"].mean()) if len(test_df) else 0.0,
    }

    # Save summary JSON files
    with open(out_path / "full_scene_spatial_summary.json", "w") as f:
        json.dump(full_scene_summary, f, indent=2)

    with open(out_path / "test_split_sanity_check.json", "w") as f:
        json.dump(test_sanity_summary, f, indent=2)

    return patch_df, full_scene_summary, test_sanity_summary, full_builtup_mask


# ---------------------------------------------------------------------------
# 6. Multi-Panel Spatial Intelligence Map Visualization
# ---------------------------------------------------------------------------

def export_spatial_hotspot_maps(
    patch_df: pd.DataFrame,
    clusters_gdf: gpd.GeoDataFrame,
    out_dir: str,
    transform: Optional[object] = None,
    patch_size: int = 256,
) -> None:
    """
    Generates spatial intelligence maps across the full study scene:
      Panel 1: Built-up expansion density (% per patch).
      Panel 2: MC Dropout model uncertainty heatmap (std deviation).
      Panel 3: Hotspot z-score surface with detected multi-patch cluster polygons overlay.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    gdf = compute_patch_geometries(patch_df.to_dict(orient="records"), transform=transform, patch_size=patch_size)
    gdf["builtup_pct"] = gdf["builtup_fraction"] * 100.0
    gdf["z_score"] = patch_df["z_score"] if "z_score" in patch_df.columns else (
        (patch_df["builtup_fraction"] - patch_df["builtup_fraction"].mean()) / max(patch_df["builtup_fraction"].std(), 1e-6)
    )
    gdf["uncertainty"] = patch_df["mean_uncertainty_change"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # Panel 1: Built-up Expansion Density
    gdf.plot(column="builtup_pct", ax=axes[0], cmap="YlOrRd", legend=True,
             legend_kwds={"label": "Built-up Expansion (% of valid area)", "orientation": "horizontal", "shrink": 0.7})
    axes[0].set_title("Built-Up Expansion Density (%)", fontsize=12, fontweight="bold")
    axes[0].set_xlabel("Longitude" if transform else "Pixel Col")
    axes[0].set_ylabel("Latitude" if transform else "Pixel Row")

    # Panel 2: Model Uncertainty
    gdf.plot(column="uncertainty", ax=axes[1], cmap="viridis", legend=True,
             legend_kwds={"label": "MC Dropout Std Deviation (Uncertainty)", "orientation": "horizontal", "shrink": 0.7})
    axes[1].set_title("Model Predictive Uncertainty", fontsize=12, fontweight="bold")
    axes[1].set_xlabel("Longitude" if transform else "Pixel Col")
    axes[1].set_ylabel("Latitude" if transform else "Pixel Row")

    # Panel 3: Hotspot Z-scores & Cluster Polygons
    gdf.plot(column="z_score", ax=axes[2], cmap="coolwarm", vmin=-2, vmax=4, legend=True,
             legend_kwds={"label": "Hotspot Z-Score", "orientation": "horizontal", "shrink": 0.7})

    if len(clusters_gdf) > 0 and not clusters_gdf.empty:
        clusters_gdf.boundary.plot(ax=axes[2], color="black", linewidth=2.5, linestyle="--", label="Hotspot Clusters")
        for _, cl_row in clusters_gdf.iterrows():
            c_poly = cl_row.geometry
            cx, cy = c_poly.centroid.x, c_poly.centroid.y
            axes[2].text(cx, cy, f"Cluster {int(cl_row['cluster_id'])}\n({cl_row['total_builtup_km2']:.1f} km²)",
                         color="black", fontsize=9, fontweight="bold", ha="center",
                         bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.8, edgecolor="black"))

    axes[2].set_title("Identified Growth Hotspots & Clusters", fontsize=12, fontweight="bold")
    axes[2].set_xlabel("Longitude" if transform else "Pixel Col")
    axes[2].set_ylabel("Latitude" if transform else "Pixel Row")

    plt.tight_layout()
    fig_path = out_path / "spatial_intelligence_hotspot_maps.png"
    fig.savefig(fig_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Exported spatial maps to {fig_path}")
