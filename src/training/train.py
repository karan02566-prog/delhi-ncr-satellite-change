"""Phase 4 baseline training loop. Checkpoints to Drive every epoch
(Colab free-tier sessions can drop without warning)."""
import torch
from torch.utils.data import DataLoader
from pathlib import Path

from src.models.baseline import BaselineChangeCNN
from src.training.losses import masked_bce_dice_loss
from src.training.metrics import compute_change_metrics


def train_baseline(train_dataset, val_dataset, checkpoint_dir: str,
                    epochs: int = 20, batch_size: int = 8, lr: float = 1e-3,
                    resume: bool = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = BaselineChangeCNN(in_channels=12).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    ckpt_path = Path(checkpoint_dir) / "baseline_latest.pt"
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    best_f1 = 0.0

    if resume and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt["epoch"] + 1
        best_f1 = ckpt.get("best_f1", 0.0)
        print(f"Resumed from epoch {start_epoch}, best_f1={best_f1:.4f}")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    

    for epoch in range(start_epoch, epochs):
        model.train()
        train_loss = 0.0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            logits = model(x)
            loss = masked_bce_dice_loss(logits, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * x.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        val_loss = 0.0
        agg = {"precision": 0.0, "recall": 0.0, "f1": 0.0, "iou": 0.0}
        n_batches = 0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x)
                loss = masked_bce_dice_loss(logits, y)
                val_loss += loss.item() * x.size(0)
                m = compute_change_metrics(logits, y)
                for k in agg:
                    agg[k] += m[k]
                n_batches += 1
        val_loss /= len(val_dataset)
        for k in agg:
            agg[k] /= n_batches

        print(f"Epoch {epoch+1}/{epochs} | train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
              f"| val_F1={agg['f1']:.4f} val_IoU={agg['iou']:.4f} "
              f"precision={agg['precision']:.4f} recall={agg['recall']:.4f}")

        torch.save({
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "best_f1": max(best_f1, agg["f1"]),
        }, ckpt_path)

        if agg["f1"] > best_f1:
            best_f1 = agg["f1"]
            torch.save({"epoch": epoch, "model_state": model.state_dict(), "val_metrics": agg},
                       Path(checkpoint_dir) / "baseline_best.pt")

    return model, best_f1
