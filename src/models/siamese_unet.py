"""Phase 5 — Siamese U-Net. Shared-weight encoder branches for T1/T2,
feature-difference fusion at each scale, U-Net decoder with skip
connections. Compare directly against BaselineChangeCNN (Phase 4)."""
import torch
import torch.nn as nn


def conv_block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
    )


class SharedEncoder(nn.Module):
    """Single shared-weight encoder, applied to both T1 and T2."""
    def __init__(self, in_channels=6):
        super().__init__()
        self.enc1 = conv_block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = conv_block(64, 128)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        return e1, e2, b


class SiameseUNet(nn.Module):
    def __init__(self, in_channels=6):
        super().__init__()
        self.encoder = SharedEncoder(in_channels)  # SAME weights applied to T1 and T2

        # Fusion: absolute difference at each scale (standard change-detection fusion)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = conv_block(128, 64)   # cat(up(b_diff), e2_diff)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = conv_block(64, 32)    # cat(up(d2), e1_diff)
        self.out_conv = nn.Conv2d(32, 1, 1)

    def forward(self, t1, t2):
        e1_a, e2_a, b_a = self.encoder(t1)
        e1_b, e2_b, b_b = self.encoder(t2)  # shared weights — same module, called twice

        e1_diff = torch.abs(e1_a - e1_b)
        e2_diff = torch.abs(e2_a - e2_b)
        b_diff = torch.abs(b_a - b_b)

        d2 = self.dec2(torch.cat([self.up2(b_diff), e2_diff], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1_diff], dim=1))
        return self.out_conv(d1).squeeze(1)