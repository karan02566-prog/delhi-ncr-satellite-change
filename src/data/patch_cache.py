"""Builds on-disk patch cache (T1, T2, label) from split manifest so
training doesn't need to reload full rasters / recompute labels each
run. Run once per session (or skip if cache already exists on Drive)."""
import json
import numpy as np
from pathlib import Path


def build_patch_cache(manifest_path: str, t1_array: np.ndarray, t2_array: np.ndarray,
                       out_dir: str, patch_size: int = 256):
    with open(manifest_path) as f:
        manifest = json.load(f)

    out = Path(out_dir)
    for split in ("train", "val", "test"):
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        items = manifest["splits"][split]
        for i, item in enumerate(items):
            r, c = item["row"], item["col"]
            fpath = split_dir / f"patch_{i:05d}_r{r}_c{c}.npz"
            if fpath.exists():
                continue
            t1p = t1_array[:, r:r + patch_size, c:c + patch_size]
            t2p = t2_array[:, r:r + patch_size, c:c + patch_size]
            np.savez_compressed(fpath, t1=t1p, t2=t2p, row=r, col=c)
        print(f"{split}: cached {len(items)} patch inputs (labels regenerated in Dataset).")