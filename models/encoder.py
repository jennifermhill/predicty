"""hybrid 3D-stem + 2D ResNet-50 encoder.

input  : (B, 1, Z, H, W)   Z is the slab depth (10 globals, 5 locals)
                            H = W is the spatial crop size (256 globals, 96 locals)
output : (B, latent_dim)    latent_dim is 2048 for ResNet-50

design:
  - a small 3D stem learns depth-aware features over the z slab.
  - ZAttentionPool collapses z using learned per-slice attention weights; one
    set of encoder weights handles both globals (Z=10) and locals (Z=5).
  - the 2D backbone is a ResNet-50 with its first conv adapted to take
    `stem_channels` inputs and its classification head removed. weights
    are randomly initialized; ImageNet pretraining doesn't transfer to
    microscopy at the scale of this dataset.
  - normalization is groupnorm, not batchnorm. groupnorm computes statistics
    per-sample over channel groups, so there are no running-stat buffers. this
    is deliberate: the ssl training loop pushes masked and multi-scale crops
    through this encoder, whose degenerate statistics would corrupt batchnorm
    running stats and then ship in the frozen eval checkpoint.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50


class ZAttentionPool(nn.Module):
    """learned z-aggregation: replaces fixed average-pool over the depth axis.

    computes one attention weight per z-slice (spatially averaged), softmaxes
    over z, then returns a weighted sum -- preserving depth structure the model
    found informative rather than discarding it uniformly.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Z, H, W)
        # average over spatial dims to get one score per z-slice
        w = self.score(x).mean(dim=(-1, -2), keepdim=True)  # (B, 1, Z, 1, 1)
        w = w.softmax(dim=2)
        return (x * w).sum(dim=2, keepdim=True)  # (B, C, 1, H, W)


class HybridEncoder(nn.Module):
    def __init__(
        self, stem_channels: int = 64, latent_dim: int = 2048, norm_groups: int = 32
    ) -> None:
        super().__init__()
        if stem_channels < 1:
            raise ValueError(f"stem_channels must be >= 1; got {stem_channels}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
        if norm_groups < 1:
            raise ValueError(f"norm_groups must be >= 1; got {norm_groups}")

        def make_norm(num_channels: int) -> nn.GroupNorm:
            # clamp groups to the largest divisor of num_channels that is
            # <= norm_groups so groupnorm is valid for any channel count.
            # groupnorm handles 3D (stem) and 2D (backbone) tensors alike.
            groups = min(norm_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, num_channels)

        # 3D stem: two conv layers that preserve (Z, H, W) and grow channels.
        # ZAttentionPool then collapses z with learned weights so the 2D
        # backbone sees a single feature plane per sample.
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            make_norm(32),
            nn.GELU(),
            nn.Conv3d(32, stem_channels, kernel_size=3, padding=1, bias=False),
            make_norm(stem_channels),
            nn.GELU(),
        )
        self.z_pool = ZAttentionPool(stem_channels)

        # 2D ResNet-50 backbone. swap the input conv (3 -> stem_channels) and
        # drop the classifier (we want the 2048-d pre-classifier feature). the
        # norm_layer factory replaces every backbone batchnorm with groupnorm.
        self.backbone = resnet50(weights=None, norm_layer=make_norm)
        self.backbone.conv1 = nn.Conv2d(
            stem_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # Keep the default checkpoint layout unchanged. For an explicitly
        # smaller/larger representation, use ResNet's final projection layer.
        self.backbone.fc = (
            nn.Identity() if latent_dim == 2048 else nn.Linear(2048, latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, Z, H, W) -> stem -> (B, stem_channels, Z, H, W)
        x = self.stem(x)
        # learned z collapse -> (B, stem_channels, 1, H, W)
        x = self.z_pool(x)
        # drop trivial z axis -> (B, stem_channels, H, W) for ResNet
        x = x.squeeze(2)
        # groupnorm retains each layer's full input for backward (unlike the
        # fused cudnn batchnorm it replaced), so checkpoint the four resnet
        # stages: recompute their activations in backward instead of storing
        # them. only during training, where the backward pass exists.
        b = self.backbone
        x = b.maxpool(b.relu(b.bn1(b.conv1(x))))
        for stage in (b.layer1, b.layer2, b.layer3, b.layer4):
            x = checkpoint(stage, x, use_reentrant=False) if self.training else stage(x)
        x = b.avgpool(x).flatten(1)
        # fc is Identity for latent_dim 2048, else a linear projection.
        return b.fc(x)


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
