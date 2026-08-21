"""PyTorch Dataset for Phase 4+ training. Reads cached T1/T2 patch
.npz files, computes the change label on-the-fly using the SAME
compute_change_labels() logic/thresholds as Phase 3 (guarantees no
label drift between phases)."""
import glob
import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.label_generation import compute_change_labels, INVALID_LABEL


class ChangeDetectionPatchDataset(Dataset):
    def __init__(self, cache_dir: str, ndvi_std_multiplier=2.2, ndbi_std_multiplier=1.5):
        self.files = sorted(glob.glob(f"{cache_dir}/*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No cached patches found in {cache_dir}")
        self.ndvi_k = ndvi_std_multiplier
        self.ndbi_k = ndbi_std_multiplier

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        t1p, t2p = data["t1"], data["t2"]

        result = compute_change_labels(t1p, t2p, self.ndvi_k, self.ndbi_k)
        label = result.label  # (H, W) uint8: 0/1/255

        x = np.concatenate([t1p, t2p], axis=0).astype(np.float32)  # (12, H, W)
        x = np.nan_to_num(x, nan=0.0)

        return torch.from_numpy(x), torch.from_numpy(label.astype(np.int64))