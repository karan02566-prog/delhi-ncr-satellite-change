"""
Phase 9 — Spatial Hotspot Detection, Area Metrics, and Urban Edge Proximity.

This module provides spatial intelligence algorithms for the Delhi NCR Urban
Change Intelligence project:
  - Real-world area conversion from Sentinel-2 10m pixel counts (m², ha, km²).
  - Data-driven adaptive cutoff for baseline (T1) urban core identification.
  - Visual verification chip export for the adaptive urban cutoff.
  - Geographic patch bounding box and centroid geometry calculation.
  - Statistical z-score filtering and DBSCAN spatial hotspot clustering.
  - Euclidean distance transform proximity analysis to the baseline urban edge.

All spatial analysis is strictly based on continuous geographic coordinates and
raster geometry without political or administrative framing.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from scipy.ndimage import distance_transform_edt, gaussian_filter
from scipy.spatial import cKDTree
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


# ---------------------------------------------------------------------------
# 1. Area conversions (Sentinel-2 10m pixel resolution)
# ---------------------------------------------------------------------------

def calculate_pixel_area(pixel_count: Union[int, np.ndarray, pd.Series],
                         pixel_res_m: float = 10.0) -> Dict[str, Union[float, np.ndarray, pd.Series]]:
    """
    Converts pixel count to real physical area units based on pixel resolution.

    Args:
        pixel_count: Number of pixels (int, numpy array, or pandas Series).
        pixel_res_m: Pixel spatial resolution in meters (default 10.0m for Sentinel-2).

    Returns:
        dict containing:
            'pixel_count': input pixel count
            'area_m2': area in square meters (m²)
            'area_ha': area in hectares (1 ha = 10,000 m²)
            'area_km2': area in square kilometers (1 km² = 1,000,000 m²)
    """
    pixel_area_m2 = pixel_res_m * pixel_res_m  # 100 m² per pixel
    area_m2 = pixel_count * pixel_area_m2
    area_ha = area_m2 / 10_000.0
    area_km2 = area_m2 / 1_000_000.0

    return {
        "pixel_count": pixel_count,
        "area_m2": area_m2,
        "area_ha": area_ha,
        "area_km2": area_km2,
    }


# ---------------------------------------------------------------------------
# 2. Adaptive Baseline Urban Core Cutoff (Data-Driven)
# ---------------------------------------------------------------------------

def derive_adaptive_urban_cutoff(t1_ndbi: np.ndarray,
                                 valid_mask: np.ndarray,
                                 std_mult: float = 1.0) -> Tuple[float, Dict[str, float]]:
    """
    Derives an adaptive data-driven NDBI threshold to delineate the pre-existing
    baseline urban core from the T1 raster, rather than hardcoding a cutoff.

    Formula:
        urban_cutoff = mean(NDBI_valid) + std_mult * std(NDBI_valid)

    Args:
        t1_ndbi: (H, W) float array of T1 NDBI values.
        valid_mask: (H, W) bool mask of valid non-cloud/finite pixels.
        std_mult: Multiplier for standard deviation (default 1.0).

    Returns:
        urban_cutoff (float): Adaptive threshold above which pixels are categorized as baseline urban.
        metadata (dict): Summary statistics (mean, std, valid pixel count, urban fraction).
    """
    valid_ndbi = t1_ndbi[valid_mask]
    if valid_ndbi.size == 0:
        raise ValueError("Cannot derive urban cutoff: valid_mask has 0 valid pixels.")

    mean_val = float(np.nanmean(valid_ndbi))
    std_val = float(np.nanstd(valid_ndbi))
    urban_cutoff = float(mean_val + std_mult * std_val)

    urban_mask = (t1_ndbi >= urban_cutoff) & valid_mask
    urban_pixel_count = int(urban_mask.sum())
    urban_fraction = float(urban_pixel_count / valid_mask.sum())

    metadata = {
        "mean_ndbi": mean_val,
        "std_ndbi": std_val,
        "std_multiplier": std_mult,
        "urban_cutoff": urban_cutoff,
        "valid_pixels": int(valid_mask.sum()),
        "urban_pixels": urban_pixel_count,
        "urban_fraction_of_valid": urban_fraction,
    }

    return urban_cutoff, metadata


def export_urban_cutoff_verification_chips(t1_array: np.ndarray,
                                           t1_ndbi: np.ndarray,
                                           valid_mask: np.ndarray,
                                           urban_cutoff: float,
                                           out_dir: str,
                                           patch_size: int = 256,
                                           n_samples: int = 8,
                                           seed: int = 42) -> List[Dict]:
    """
    Exports visual verification image chips showing:
      1. T1 True-Color RGB
      2. T1 NDBI continuous heatmap
      3. Derived Urban Core Mask overlay (magenta) over T1 RGB

    Stratified across low/medium/high urban fraction patches to sanity-check
    the adaptive cutoff across dense urban, rural-urban fringe, and rural areas.

    Args:
        t1_array: (6, H, W) T1 reflectance array (B2, B3, B4, B8, B11, B12).
        t1_ndbi: (H, W) float array of T1 NDBI.
        valid_mask: (H, W) bool array.
        urban_cutoff: Adaptive NDBI threshold.
        out_dir: Output directory path.
        patch_size: Square chip size in pixels (default 256).
        n_samples: Number of sample chips to export (default 8).
        seed: Random seed for stratified selection.

    Returns:
        List of metadata dicts for exported verification chips.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    urban_mask = (t1_ndbi >= urban_cutoff) & valid_mask
    _, H, W = t1_array.shape

    # Generate candidate grid chips
    grid_chips = []
    for r in range(0, H - patch_size + 1, patch_size):
        for c in range(0, W - patch_size + 1, patch_size):
            v_patch = valid_mask[r:r + patch_size, c:c + patch_size]
            if v_patch.mean() < 0.5:
                continue
            u_patch = urban_mask[r:r + patch_size, c:c + patch_size]
            u_frac = float(u_patch[v_patch].mean()) if v_patch.any() else 0.0
            grid_chips.append({"row": r, "col": c, "urban_fraction": u_frac})

    if not grid_chips:
        return []

    # Stratify into low, medium, high urban density
    df = pd.DataFrame(grid_chips)
    df["stratum"] = pd.qcut(df["urban_fraction"], q=3, labels=["low", "medium", "high"], duplicates="drop")

    rng = np.random.default_rng(seed)
    sampled_indices = []
    per_stratum = max(1, n_samples // len(df["stratum"].unique()))
    for stratum_name, grp in df.groupby("stratum", observed=False):
        n_pick = min(per_stratum, len(grp))
        chosen = rng.choice(grp.index, size=n_pick, replace=False)
        sampled_indices.extend(chosen)

    sampled_df = df.loc[sampled_indices].head(n_samples)

    r_idx, g_idx, b_idx = 2, 1, 0  # B4 (red), B3 (green), B2 (blue)
    exported_records = []

    for i, (_, row_item) in enumerate(sampled_df.iterrows()):
        r = int(row_item["row"])
        c = int(row_item["col"])
        u_frac = float(row_item["urban_fraction"])
        stratum_val = str(row_item["stratum"])

        t1_chip = t1_array[[r_idx, g_idx, b_idx], r:r + patch_size, c:c + patch_size]
        t1_chip = np.transpose(t1_chip, (1, 2, 0))
        t1_chip = np.nan_to_num(t1_chip, nan=0.0)
        p2, p98 = np.percentile(t1_chip, [2, 98])
        t1_rgb = np.clip((t1_chip - p2) / max(p98 - p2, 1e-6), 0, 1)

        ndbi_chip = t1_ndbi[r:r + patch_size, c:c + patch_size]
        u_mask_chip = urban_mask[r:r + patch_size, c:c + patch_size]

        overlay = t1_rgb.copy()
        overlay[u_mask_chip] = [0.9, 0.1, 0.8]  # magenta for baseline urban core

        fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
        axes[0].imshow(t1_rgb)
        axes[0].set_title(f"T1 RGB (row {r}, col {c})")
        axes[0].axis("off")

        im1 = axes[1].imshow(ndbi_chip, cmap="coolwarm", vmin=-0.4, vmax=0.4)
        axes[1].set_title(f"T1 NDBI (cutoff = {urban_cutoff:.3f})")
        axes[1].axis("off")
        plt.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

        axes[2].imshow(overlay)
        axes[2].set_title(f"Urban Core Overlay ({stratum_val}, {u_frac*100:.1f}%)")
        axes[2].axis("off")

        plt.tight_layout()
        fname = f"urban_cutoff_chip_{i:02d}_{stratum_val}_r{r}_c{c}.png"
        fpath = out_path / fname
        fig.savefig(fpath, dpi=120)
        plt.close(fig)

        exported_records.append({
            "chip_index": i,
            "filename": fname,
            "row": r,
            "col": c,
            "stratum": stratum_val,
            "urban_fraction": round(u_frac, 4),
            "urban_cutoff": round(urban_cutoff, 4),
        })

    # Save summary verification log
    pd.DataFrame(exported_records).to_csv(out_path / "urban_cutoff_verification_log.csv", index=False)
    return exported_records


# ---------------------------------------------------------------------------
# 3. Patch Georeferencing
# ---------------------------------------------------------------------------

def compute_patch_geometries(patch_records: List[Dict],
                             transform: Optional[object] = None,
                             patch_size: int = 256) -> gpd.GeoDataFrame:
    """
    Computes EPSG:4326 polygon bounding boxes and centroid coordinates for
    each patch given its row/col offsets and affine geotransform.

    Args:
        patch_records: List of dicts with 'row' and 'col' keys.
        transform: rasterio.Affine transform. If None, pixel-space coordinates are used.
        patch_size: Square patch side length in pixels (default 256).

    Returns:
        gpd.GeoDataFrame containing patch metadata, centroid points, and polygon geometries.
    """
    geoms = []
    centroids_x = []
    centroids_y = []

    for item in patch_records:
        r, c = item["row"], item["col"]
        min_row, max_row = r, r + patch_size
        min_col, max_col = c, c + patch_size

        if transform is not None:
            # Transform pixel corners to geographic coordinates (lon, lat)
            x_min, y_max = transform * (min_col, min_row)
            x_max, y_min = transform * (max_col, max_row)
            cx, cy = transform * (min_col + patch_size / 2.0, min_row + patch_size / 2.0)
            poly = box(x_min, y_min, x_max, y_max)
        else:
            # Fallback to raster pixel indices
            cx, cy = c + patch_size / 2.0, r + patch_size / 2.0
            poly = box(min_col, min_row, max_col, max_row)

        geoms.append(poly)
        centroids_x.append(cx)
        centroids_y.append(cy)

    df = pd.DataFrame(patch_records)
    df["centroid_x"] = centroids_x
    df["centroid_y"] = centroids_y

    crs = "EPSG:4326" if transform is not None else None
    gdf = gpd.GeoDataFrame(df, geometry=geoms, crs=crs)
    return gdf


# ---------------------------------------------------------------------------
# 4. Spatial Hotspot Detection & DBSCAN Clustering
# ---------------------------------------------------------------------------

def detect_spatial_hotspots(patch_gdf: gpd.GeoDataFrame,
                            z_threshold: float = 1.645,
                            patch_stride_km: float = 2.56,
                            min_samples: int = 2) -> Tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, Dict]:
    """
    Detects spatial hotspots of rapid built-up expansion using statistical
    z-score filtering and DBSCAN density clustering.

    Threshold Justifications:
      - z_threshold = 1.645: Standard parametric threshold for upper 95% one-tailed
        confidence level of elevated built-up growth density.
      - eps_deg = 1.5 * patch_stride_km / 104 km/deg: Derived from physical patch grid
        spacing (~2.56 km) to connect adjacent 8-neighbor high-growth tiles.
      - min_samples = 2: Minimum cluster size to form a contiguous multi-patch
        growth corridor (single isolated patches remain localized point hotspots).

    Args:
        patch_gdf: GeoDataFrame containing 'builtup_fraction' (or 'builtup_pixels' & 'valid_pixels'),
                   'centroid_x', 'centroid_y', and polygon geometries.
        z_threshold: Z-score cutoff for candidate hotspot patches (default 1.645).
        patch_stride_km: Physical distance between adjacent patch origins (default 2.56 km).
        min_samples: Minimum candidate patches to form a cluster (default 2).

    Returns:
        candidate_gdf: GeoDataFrame of all patches with z_score >= z_threshold (with cluster_id).
        clusters_gdf: GeoDataFrame of merged cluster polygon boundaries and aggregate stats.
        summary: Metadata dict describing parameters and cluster counts.
    """
    gdf = patch_gdf.copy()

    if "builtup_fraction" not in gdf.columns:
        if "builtup_pixels" in gdf.columns and "valid_pixels" in gdf.columns:
            gdf["builtup_fraction"] = gdf["builtup_pixels"] / gdf["valid_pixels"].clip(lower=1)
        else:
            raise KeyError("patch_gdf must contain 'builtup_fraction' or 'builtup_pixels'/'valid_pixels'.")

    densities = gdf["builtup_fraction"].to_numpy()
    mean_density = float(np.mean(densities))
    std_density = float(np.std(densities))

    if std_density < 1e-8:
        gdf["z_score"] = 0.0
    else:
        gdf["z_score"] = (densities - mean_density) / std_density

    # Filter candidate patches crossing statistical significance
    candidate_mask = gdf["z_score"] >= z_threshold
    candidate_gdf = gdf[candidate_mask].copy()

    # Approximate 1 degree in km at Delhi latitude (28.6°N):
    # 1 deg lat ~ 110.8 km, 1 deg lon ~ 97.7 km -> avg ~ 104 km/deg
    km_per_deg = 104.0
    eps_deg = (1.5 * patch_stride_km) / km_per_deg  # ~0.0369 degrees

    cluster_records = []
    if len(candidate_gdf) >= min_samples:
        coords = np.column_stack([candidate_gdf["centroid_x"], candidate_gdf["centroid_y"]])
        clustering = DBSCAN(eps=eps_deg, min_samples=min_samples).fit(coords)
        candidate_gdf["cluster_id"] = clustering.labels_

        unique_clusters = [cid for cid in np.unique(clustering.labels_) if cid != -1]
        for cid in unique_clusters:
            c_patches = candidate_gdf[candidate_gdf["cluster_id"] == cid]
            geom_series = c_patches.geometry
            merged_poly = geom_series.union_all() if hasattr(geom_series, "union_all") else geom_series.unary_union
            hull_poly = merged_poly.convex_hull if hasattr(merged_poly, "convex_hull") else merged_poly

            total_builtup_px = int(c_patches["builtup_pixels"].sum()) if "builtup_pixels" in c_patches.columns else 0
            area_info = calculate_pixel_area(total_builtup_px)

            cluster_records.append({
                "cluster_id": int(cid),
                "patch_count": len(c_patches),
                "mean_z_score": float(c_patches["z_score"].mean()),
                "max_z_score": float(c_patches["z_score"].max()),
                "mean_builtup_fraction": float(c_patches["builtup_fraction"].mean()),
                "total_builtup_pixels": total_builtup_px,
                "total_builtup_km2": float(area_info["area_km2"]),
                "geometry": hull_poly,
            })
    else:
        candidate_gdf["cluster_id"] = -1

    crs = patch_gdf.crs
    if cluster_records:
        clusters_gdf = gpd.GeoDataFrame(cluster_records, crs=crs)
    else:
        clusters_gdf = gpd.GeoDataFrame(
            columns=["cluster_id", "patch_count", "mean_z_score", "max_z_score",
                     "mean_builtup_fraction", "total_builtup_pixels", "total_builtup_km2", "geometry"],
            crs=crs,
        )

    summary = {
        "z_threshold": z_threshold,
        "mean_builtup_density": mean_density,
        "std_builtup_density": std_density,
        "total_patches_evaluated": len(gdf),
        "candidate_hotspot_patches": int(len(candidate_gdf)),
        "eps_deg": eps_deg,
        "min_samples": min_samples,
        "identified_multi_patch_clusters": len(cluster_records),
        "isolated_point_hotspots": int((candidate_gdf["cluster_id"] == -1).sum()) if len(candidate_gdf) > 0 else 0,
    }

    return candidate_gdf, clusters_gdf, summary


# ---------------------------------------------------------------------------
# 5. Proximity Analysis to Baseline Urban Edge (Euclidean Distance Transform)
# ---------------------------------------------------------------------------

def compute_urban_edge_proximity(t1_ndbi: np.ndarray,
                                 builtup_change_mask: np.ndarray,
                                 urban_cutoff: float,
                                 valid_mask: Optional[np.ndarray] = None,
                                 pixel_res_m: float = 10.0) -> Dict[str, Union[float, np.ndarray]]:
    """
    Computes exact Euclidean distance from newly expanded built-up pixels to the
    nearest pre-existing baseline (T1) urban core pixel using scipy.ndimage.distance_transform_edt.

    Args:
        t1_ndbi: (H, W) float array of T1 NDBI.
        builtup_change_mask: (H, W) bool array of predicted built-up expansion pixels.
        urban_cutoff: NDBI threshold defining baseline urban core.
        valid_mask: Optional (H, W) bool mask of valid pixels.
        pixel_res_m: Spatial resolution in meters (default 10.0m).

    Returns:
        dict containing:
            'mean_distance_m': Mean distance of new built-up pixels to urban core (meters).
            'median_distance_m': Median distance (meters).
            'p25_distance_m', 'p75_distance_m', 'p90_distance_m': Distance percentiles.
            'fraction_within_500m': Fraction of new built-up pixels within 500m of existing urban.
            'fraction_within_1000m': Fraction within 1km.
            'fraction_beyond_2000m': Fraction beyond 2km (isolated / leapfrog development).
            'total_builtup_pixels_analyzed': Total pixels evaluated.
    """
    if valid_mask is None:
        valid_mask = np.ones_like(builtup_change_mask, dtype=bool)

    urban_core = (t1_ndbi >= urban_cutoff) & valid_mask

    if not urban_core.any():
        return {
            "mean_distance_m": float("nan"),
            "median_distance_m": float("nan"),
            "fraction_within_500m": 0.0,
            "fraction_within_1000m": 0.0,
            "fraction_beyond_2000m": 0.0,
            "total_builtup_pixels_analyzed": 0,
        }

    dist_pixels = distance_transform_edt(~urban_core)
    dist_meters = dist_pixels * pixel_res_m

    target_mask = builtup_change_mask & valid_mask
    if not target_mask.any():
        return {
            "mean_distance_m": 0.0,
            "median_distance_m": 0.0,
            "p25_distance_m": 0.0,
            "p75_distance_m": 0.0,
            "p90_distance_m": 0.0,
            "fraction_within_500m": 0.0,
            "fraction_within_1000m": 0.0,
            "fraction_beyond_2000m": 0.0,
            "total_builtup_pixels_analyzed": 0,
        }

    distances = dist_meters[target_mask]

    p25, p50, p75, p90 = np.percentile(distances, [25, 50, 75, 90])
    mean_dist = float(np.mean(distances))

    w500 = float((distances <= 500.0).mean())
    w1000 = float((distances <= 1000.0).mean())
    b2000 = float((distances > 2000.0).mean())

    return {
        "mean_distance_m": mean_dist,
        "median_distance_m": float(p50),
        "p25_distance_m": float(p25),
        "p75_distance_m": float(p75),
        "p90_distance_m": float(p90),
        "fraction_within_500m": w500,
        "fraction_within_1000m": w1000,
        "fraction_beyond_2000m": b2000,
        "total_builtup_pixels_analyzed": int(target_mask.sum()),
    }

