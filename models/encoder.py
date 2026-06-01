"""hybrid 3D-stem + 2D ResNet-50 encoder.

input  : (B, 1, Z, H, W)   Z is the slab depth (10 globals, 5 locals)
                            H = W is the spatial crop size (256 globals, 96 locals)
output : (B, latent_dim)    latent_dim is 2048 for ResNet-50

design:
  - a small 3D stem learns depth-aware features over the z slab.
  - AdaptiveAvgPool3D collapses z to 1 regardless of input z. one set of
    encoder weights then handles globals (Z=10) and locals (Z=5).
  - the 2D backbone is a ResNet-50 with its first conv adapted to take
    `stem_channels` inputs and its classification head removed. weights
    are randomly initialized; ImageNet pretraining doesn't transfer to
    microscopy at the scale of this dataset.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import resnet50


class HybridEncoder(nn.Module):
    def __init__(self, stem_channels: int = 64) -> None:
        super().__init__()

        # 3D stem: two conv layers that preserve (Z, H, W) and grow channels.
        # adaptive pool then collapses z to 1 so the 2D backbone sees a single
        # feature plane per sample.
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, stem_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(stem_channels),
            nn.GELU(),
            nn.AdaptiveAvgPool3d((1, None, None)),
        )

        # 2D ResNet-50 backbone. swap the input conv (3 -> stem_channels) and
        # drop the classifier (we want the 2048-d pre-classifier feature).
        self.backbone = resnet50(weights=None)
        self.backbone.conv1 = nn.Conv2d(
            stem_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.backbone.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, Z, H, W) -> stem -> (B, stem_channels, 1, H, W)
        x = self.stem(x)
        # collapse the trivial z axis -> (B, stem_channels, H, W) for ResNet.
        x = x.squeeze(2)
        # backbone returns (B, 2048).
        return self.backbone(x)


if __name__ == "__main__":
    # smoke test: verify the encoder accepts both global and local shapes
    # without needing any zarr files. run as: python -m models.encoder
    enc = HybridEncoder()
    enc.eval()
    with torch.no_grad():
        g = torch.randn(2, 1, 10, 256, 256)
        l = torch.randn(2, 1, 5, 96, 96)
        z_g = enc(g)
        z_l = enc(l)
    print(f"global input  {tuple(g.shape)} -> output {tuple(z_g.shape)}")
    print(f"local  input  {tuple(l.shape)} -> output {tuple(z_l.shape)}")
    assert z_g.shape == (2, 2048), f"unexpected global output shape: {z_g.shape}"
    assert z_l.shape == (2, 2048), f"unexpected local output shape: {z_l.shape}"
    print("encoder smoke test passed.")
