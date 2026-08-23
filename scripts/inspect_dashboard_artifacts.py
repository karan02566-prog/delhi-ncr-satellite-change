"""
One-time diagnostic script for Phase 10.
Prints the real schema of every Phase 9 artifact so the dashboard
can be built against actual column/key names instead of assumptions.
Not part of the pipeline - safe to delete after Phase 10 setup.
"""

import json
import os

import numpy as np
import pandas as pd

DATA_DIR = os.path.join("data", "dashboard", "spatial")


def inspect_csv(filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*70}\nCSV: {filename}\n{'='*70}")
    if not os.path.exists(path):
        print("  MISSING")
        return
    df = pd.read_csv(path)
    print(f"  shape: {df.shape}")
    print(f"  columns + dtypes:")
    for col, dtype in df.dtypes.items():
        print(f"    {col:35s} {dtype}")
    print(f"\n  head(3):")
    print(df.head(3).to_string())


def inspect_json(filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*70}\nJSON: {filename}\n{'='*70}")
    if not os.path.exists(path):
        print("  MISSING")
        return
    with open(path, "r") as f:
        data = json.load(f)
    print(json.dumps(data, indent=2)[:3000])


def inspect_npz(filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*70}\nNPZ: {filename}\n{'='*70}")
    if not os.path.exists(path):
        print("  MISSING")
        return
    size_mb = os.path.getsize(path) / (1024 * 1024)
    print(f"  file size: {size_mb:.1f} MB")
    npz = np.load(path, allow_pickle=True)
    for key in npz.files:
        arr = npz[key]
        print(f"    key='{key:25s}' shape={arr.shape} dtype={arr.dtype}")


def inspect_png(filename: str) -> None:
    path = os.path.join(DATA_DIR, filename)
    print(f"\n{'='*70}\nPNG: {filename}\n{'='*70}")
    if not os.path.exists(path):
        print("  MISSING")
        return
    from PIL import Image
    with Image.open(path) as img:
        print(f"  size: {img.size}, mode: {img.mode}")


if __name__ == "__main__":
    inspect_csv("patch_spatial_metrics.csv")
    inspect_json("full_scene_spatial_summary.json")
    inspect_json("test_split_sanity_check.json")
    inspect_json("urban_proximity_summary.json")
    inspect_png("spatial_intelligence_hotspot_maps.png")
    inspect_npz("full_scene_masks.npz")
    print(f"\n{'='*70}\nDONE\n{'='*70}")
