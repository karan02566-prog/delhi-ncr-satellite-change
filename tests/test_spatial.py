"""
Unit tests for Phase 9 Spatial Intelligence modules (hotspots.py, spatial_analysis.py).
"""
import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin

from src.spatial.hotspots import (
    calculate_pixel_area,
    derive_adaptive_urban_cutoff,
    compute_patch_geometries,
    detect_spatial_hotspots,
    compute_urban_edge_proximity,
)
from src.spatial.spatial_analysis import FullScenePatchDataset


def test_calculate_pixel_area_single_int():
    # 10,000 pixels at 10m resolution = 10,000 * 100 m² = 1,000,000 m² = 1 km² = 100 ha
    res = calculate_pixel_area(10000, pixel_res_m=10.0)
    assert res["pixel_count"] == 10000
    assert pytest.approx(res["area_m2"], 1e-3) == 1_000_000.0
    assert pytest.approx(res["area_ha"], 1e-3) == 100.0
    assert pytest.approx(res["area_km2"], 1e-3) == 1.0


def test_calculate_pixel_area_array():
    counts = np.array([0, 100, 5000])
    res = calculate_pixel_area(counts, pixel_res_m=10.0)
    assert np.allclose(res["area_km2"], np.array([0.0, 0.01, 0.5]))


def test_derive_adaptive_urban_cutoff():
    H, W = 100, 100
    ndbi = np.zeros((H, W), dtype=np.float32)
    ndbi[:50, :] = 0.2   # urban core region
    ndbi[50:, :] = -0.2  # non-urban region
    valid_mask = np.ones((H, W), dtype=bool)

    cutoff, meta = derive_adaptive_urban_cutoff(ndbi, valid_mask, std_mult=0.0)
    assert pytest.approx(cutoff, 1e-3) == 0.0
    assert meta["valid_pixels"] == 10000
    assert meta["urban_pixels"] == 5000
    assert pytest.approx(meta["urban_fraction_of_valid"], 1e-3) == 0.5


def test_compute_patch_geometries():
    transform = from_origin(77.0, 28.5, 0.0001, 0.0001)
    records = [
        {"row": 0, "col": 0, "split": "train"},
        {"row": 256, "col": 256, "split": "test"},
    ]
    gdf = compute_patch_geometries(records, transform=transform, patch_size=256)
    assert len(gdf) == 2
    assert gdf.crs == "EPSG:4326"
    assert isinstance(gdf.geometry.iloc[0], Polygon)
    assert "centroid_x" in gdf.columns
    assert "centroid_y" in gdf.columns


def test_detect_spatial_hotspots():
    # Construct 10 patches, 3 contiguous patches with high built-up expansion (z >= 1.645)
    records = []
    for i in range(10):
        frac = 0.40 if i in [0, 1, 2] else 0.02
        records.append({
            "patch_index": i,
            "row": (i // 5) * 256,
            "col": (i % 5) * 256,
            "builtup_fraction": frac,
            "builtup_pixels": int(frac * 65536),
            "valid_pixels": 65536,
        })
    transform = from_origin(77.0, 28.5, 0.0001, 0.0001)
    gdf = compute_patch_geometries(records, transform=transform, patch_size=256)

    candidates, clusters, summary = detect_spatial_hotspots(
        gdf,
        z_threshold=1.0,
        patch_stride_km=2.56,
        min_samples=2,
    )
    assert len(candidates) >= 3
    assert len(clusters) >= 1
    assert summary["total_patches_evaluated"] == 10


def test_compute_urban_edge_proximity():
    H, W = 100, 100
    t1_ndbi = np.full((H, W), -0.2, dtype=np.float32)
    # Urban core in top-left 10x10
    t1_ndbi[:10, :10] = 0.3

    # New built-up expansion in pixels at (20, 20) -> ~14 pixels away = ~140m
    builtup_mask = np.zeros((H, W), dtype=bool)
    builtup_mask[20, 20] = True
    valid_mask = np.ones((H, W), dtype=bool)

    res = compute_urban_edge_proximity(
        t1_ndbi=t1_ndbi,
        builtup_change_mask=builtup_mask,
        urban_cutoff=0.1,
        valid_mask=valid_mask,
        pixel_res_m=10.0,
    )
    assert res["total_builtup_pixels_analyzed"] == 1
    assert res["mean_distance_m"] > 0
    assert res["fraction_within_500m"] == 1.0


def test_full_scene_patch_dataset_slicing():
    t1 = np.ones((6, 512, 512), dtype=np.float32)
    t2 = np.ones((6, 512, 512), dtype=np.float32)
    records = [{"row": 0, "col": 0, "split": "train"}, {"row": 256, "col": 256, "split": "val"}]

    ds = FullScenePatchDataset(patch_records=records, t1_array=t1, t2_array=t2, patch_size=256)
    assert len(ds) == 2
    x, t1p, t2p, meta = ds[0]
    assert x.shape == (12, 256, 256)
    assert t1p.shape == (6, 256, 256)
    assert t2p.shape == (6, 256, 256)
    assert meta["split"] == "train"

