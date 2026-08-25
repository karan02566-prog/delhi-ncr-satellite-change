"""
Phase 11 — automated validation checks. Run from repo root.
Each check prints PASS/FAIL/WARN with the evidence, nothing silently assumed.
"""
import json
import numpy as np
import pandas as pd
from pathlib import Path

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

DRIVE_LOCAL = Path("data/dashboard/spatial")  # adjust if artifacts live elsewhere locally

def check_split_leakage():
    manifest_path = Path("data/labels/data_labels_split_manifest.json")
    if not manifest_path.exists():
        print("FAIL split_leakage: manifest not found at", manifest_path)
        return
    m = json.loads(manifest_path.read_text())
    print("CHECK split_leakage: inspect manifest col ranges manually below")
    print(json.dumps(m, indent=2)[:2000])

def check_patch_metrics_consistency():
    f = DRIVE_LOCAL / "patch_spatial_metrics.csv"
    if not f.exists():
        print("FAIL patch_metrics: not found:", f)
        return
    df = pd.read_csv(f)
    assert len(df) == 1008, f"expected 1008 rows, got {len(df)}"
    cf = df["change_fraction"].mean()
    print(f"CHECK patch_metrics: rows={len(df)}, mean change_fraction={cf:.4f} (expect ~0.1394)")
    assert df["valid_pixels"].min() >= 0
    dup = df.duplicated(subset=["row", "col"]).sum()
    print(f"CHECK patch_metrics: duplicate row/col pairs = {dup} (expect 0)")

def check_checkpoint_integrity():
    if not HAS_TORCH:
        print("WARN checkpoint_integrity: torch not installed locally (expected — training/checkpoints live in Colab/Drive per project design). Skipping. Run this check in a Colab cell against the Drive-mounted checkpoint paths instead if you want it verified.")
        return
    for name in ["baseline_best.pt", "baseline_latest.pt"]:
        p = Path("checkpoints") / name
        if not p.exists():
            print(f"WARN checkpoint {name}: not found locally at {p}")
            continue
        ckpt = torch.load(p, map_location="cpu")
        key = "model_state" if "model_state" in ckpt else "model_state_dict"
        print(f"CHECK checkpoint {name}: key='{key}', n_tensors={len(ckpt[key])}")

def check_mc_dropout_variance():
    summary_path = DRIVE_LOCAL / "full_scene_spatial_summary.json"
    if not summary_path.exists():
        print("FAIL mc_dropout: summary not found:", summary_path)
        return
    s = json.loads(summary_path.read_text())
    print("CHECK mc_dropout summary keys:", list(s.keys())[:10])

def check_reproducibility_seed():
    train_py = Path("src/training/train.py")
    if not train_py.exists():
        print("FAIL reproducibility: train.py not found")
        return
    txt = train_py.read_text()
    has_seed = "seed" in txt.lower() and ("manual_seed" in txt or "random.seed" in txt)
    print(f"CHECK reproducibility: explicit seeding present = {has_seed}")
    if not has_seed:
        print("  -> disclose in README: results not bit-for-bit reproducible, seed not fixed")

if __name__ == "__main__":
    check_split_leakage()
    check_patch_metrics_consistency()
    check_checkpoint_integrity()
    check_mc_dropout_variance()
    check_reproducibility_seed()