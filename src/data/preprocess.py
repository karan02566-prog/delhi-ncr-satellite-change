"""
Local (non-GEE) preprocessing utilities for the Delhi NCR Urban Change
Intelligence project. These functions operate on downloaded GeoTIFFs
using rasterio/numpy, and are meant to run either locally or in Colab
after the cloud-masked composites have been exported from Earth Engine.

Responsibilities covered here:
    - Loading and inspecting raster metadata (CRS, resolution, shape)
    - Validating spatial alignment between two rasters
    - Band normalization
    - Patch generation for model training (used starting Phase 3+,
      once labels exist)

No fabricated statistics: every function here returns values computed
directly from the actual raster data passed in.
"""

import numpy as np
import rasterio


def load_raster_info(path):
    """
    Open a raster and return its key metadata without loading full
    pixel data into memory (cheap inspection step).

    Args:
        path (str): path to a GeoTIFF file.

    Returns:
        dict with 'crs', 'transform', 'width', 'height', 'count',
        'dtype', 'bounds', 'nodata'.
    """
    with rasterio.open(path) as src:
        return {
            "crs": str(src.crs),
            "transform": src.transform,
            "width": src.width,
            "height": src.height,
            "count": src.count,
            "dtype": src.dtypes[0],
            "bounds": src.bounds,
            "nodata": src.nodata,
        }


def validate_alignment(path_a, path_b):
    """
    Check whether two rasters are spatially aligned: same CRS, same
    pixel dimensions, same geotransform. This must pass before any
    pixel-wise comparison (e.g. change detection) between T1 and T2
    is valid — otherwise we'd be comparing misaligned pixels.

    Args:
        path_a (str): path to first raster (e.g. T1)
        path_b (str): path to second raster (e.g. T2)

    Returns:
        dict with 'aligned' (bool) and a breakdown of which specific
        checks passed/failed, so a failure is diagnosable rather than
        just a flat "no".
    """
    info_a = load_raster_info(path_a)
    info_b = load_raster_info(path_b)

    checks = {
        "same_crs": info_a["crs"] == info_b["crs"],
        "same_width": info_a["width"] == info_b["width"],
        "same_height": info_a["height"] == info_b["height"],
        "same_transform": info_a["transform"] == info_b["transform"],
    }
    checks["aligned"] = all(checks.values())

    return {
        "aligned": checks["aligned"],
        "checks": checks,
        "info_a": info_a,
        "info_b": info_b,
    }


def normalize_bands(array, method="percentile", low_pct=2, high_pct=98):
    """
    Normalize a multi-band array to [0, 1] range, band-by-band.

    Args:
        array (np.ndarray): shape (bands, height, width)
        method (str): 'percentile' (robust to outliers, recommended
            for satellite imagery with occasional extreme values) or
            'minmax' (simple, sensitive to outliers).
        low_pct, high_pct (float): percentile clip bounds, only used
            when method='percentile'.

    Returns:
        np.ndarray: same shape, values clipped to [0, 1], dtype float32.
    """
    normalized = np.zeros_like(array, dtype=np.float32)

    for b in range(array.shape[0]):
        band = array[b].astype(np.float32)
        valid = band[~np.isnan(band)]

        if valid.size == 0:
            normalized[b] = 0
            continue

        if method == "percentile":
            lo, hi = np.percentile(valid, [low_pct, high_pct])
        else:  # minmax
            lo, hi = valid.min(), valid.max()

        if hi == lo:
            normalized[b] = 0
        else:
            normalized[b] = np.clip((band - lo) / (hi - lo), 0, 1)

    return normalized


def generate_patches(array, patch_size=256, stride=None):
    """
    Slice a full raster array into fixed-size square patches for model
    training. Patches at the right/bottom edges that don't fit evenly
    are dropped (not padded) to avoid introducing artificial nodata
    regions into training data.

    Args:
        array (np.ndarray): shape (bands, height, width)
        patch_size (int): patch side length in pixels
        stride (int or None): step size between patches; defaults to
            patch_size (non-overlapping). Use stride < patch_size for
            overlapping patches (more training data, more redundancy).

    Returns:
        list of dicts, each with:
            'patch': np.ndarray (bands, patch_size, patch_size)
            'row': top-left row index in the source array
            'col': top-left col index in the source array
        This row/col metadata is essential later for geographically
        separated train/val/test splitting (Phase 3) — patches from
        the same neighborhood must not leak across splits.
    """
    if stride is None:
        stride = patch_size

    _, height, width = array.shape
    patches = []

    for row in range(0, height - patch_size + 1, stride):
        for col in range(0, width - patch_size + 1, stride):
            patch = array[:, row : row + patch_size, col : col + patch_size]
            patches.append({"patch": patch, "row": row, "col": col})

    return patches