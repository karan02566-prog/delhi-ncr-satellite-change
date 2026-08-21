"""Phase 4 baseline: shallow FCN over concatenated T1+T2 (12-channel
input). Establishes a performance floor before the Siamese U-Net."""
import torch
import torch.nn as nn


class BaselineChangeCNN(nn.Module):
    def __init__(self, in_channels: int = 12):
        super().__init__()

        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                nn.Conv2d(cout, cout, 3, padding=1), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            )

        self.enc1 = block(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = block(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.bottleneck = block(64, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = block(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = block(64, 32)
        self.out_conv = nn.Conv2d(32, 1, 1)  # binary logit

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        b = self.bottleneck(self.pool2(e2))
        d2 = self.dec2(torch.cat([self.up2(b), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))
        return self.out_conv(d1).squeeze(1)  # (B, H, W) logits