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
        self,
        stem_channels: int = 64,
        latent_dim: int = 2048,
        norm_groups: int = 32,
        chunk_size: int | None = None,
        checkpoint_stem: bool = True,
        checkpoint_backbone: bool = False,
    ) -> None:
        super().__init__()
        if stem_channels < 1:
            raise ValueError(f"stem_channels must be >= 1; got {stem_channels}")
        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1; got {latent_dim}")
        if norm_groups < 1:
            raise ValueError(f"norm_groups must be >= 1; got {norm_groups}")
        if chunk_size is not None and chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1 or None; got {chunk_size}")

        # the dino wrapper flattens all crops into one batch (e.g. batch_size *
        # 2 globals), then pushes them through the 3D stem at full resolution.
        # the stem's activations dominate memory (a 64-channel 256x256x10 map is
        # ~20 GiB in fp32 for 128 crops), so cap how many crops flow through the
        # stem at once. None means no chunking.
        self.chunk_size = chunk_size

        # gradient checkpointing is split into two independently-controlled
        # halves because their cost/benefit is wildly different.
        #
        #   stem + z_pool: activations are 3D and full-resolution
        #     (stem_channels x Z x H x W per crop), so they dominate memory.
        #     recomputing them is cheap relative to what they save -- keep on.
        #   2D backbone: activations are post-z_pool and much smaller, while a
        #     ResNet-50 forward is most of the flops. checkpointing it pays a
        #     second full forward for a modest memory saving, which is why the
        #     old whole-forward checkpointing cost ~25-33% throughput.
        #
        # default is therefore stem-only. flip checkpoint_backbone back on if a
        # larger batch/crop count ooms.
        self.checkpoint_stem = bool(checkpoint_stem)
        self.checkpoint_backbone = bool(checkpoint_backbone)

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

    # ---- pieces of the forward, so checkpointing can be applied per-half ----

    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 1, Z, H, W) -> stem -> (B, stem_channels, Z, H, W)
        x = self.stem(x)
        # learned z collapse -> (B, stem_channels, 1, H, W)
        x = self.z_pool(x)
        # drop trivial z axis -> (B, stem_channels, H, W) for ResNet
        return x.squeeze(2)

    def _backbone_pooled(self, x: torch.Tensor) -> torch.Tensor:
        # (B, latent_dim); fc is Identity when latent_dim == 2048.
        return self.backbone(x)

    def _backbone_tokens(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x)
        if layer == "layer1":
            return x
        x = b.layer2(x)
        if layer == "layer2":
            return x
        x = b.layer3(x)
        if layer == "layer3":
            return x
        return b.layer4(x)

    def _backbone_pooled_and_tokens(
        self, x: torch.Tensor, layer: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """one backbone pass yielding both the pooled feature and `layer` tokens.

        the callers that need both (the iBOT teacher: pooled feature for the DINO
        target, patch tokens for the masked-prediction target) used to run the
        whole stem+backbone twice on the same input.
        """
        b = self.backbone
        x = b.conv1(x); x = b.bn1(x); x = b.relu(x); x = b.maxpool(x)
        x = b.layer1(x)
        tok = x if layer == "layer1" else None
        x = b.layer2(x)
        if tok is None and layer == "layer2":
            tok = x
        x = b.layer3(x)
        if tok is None and layer == "layer3":
            tok = x
        x = b.layer4(x)
        if tok is None:  # "layer4" (and any unrecognized value, as before)
            tok = x
        z = torch.flatten(b.avgpool(x), 1)
        return b.fc(z), tok

    def _ckpt_stem(self, x: torch.Tensor) -> torch.Tensor:
        # groupnorm retains its activations (it recomputes nothing in backward,
        # unlike the fused cudnn batchnorm it replaced), and the stem's are 3D at
        # full resolution, so this is where checkpointing earns its keep.
        # training only: there is no backward pass to trade against at inference.
        if self.training and self.checkpoint_stem:
            return checkpoint(self._stem_forward, x, use_reentrant=False)
        return self._stem_forward(x)

    def _ckpt_backbone(self, fn, *args):
        if self.training and self.checkpoint_backbone:
            return checkpoint(fn, *args, use_reentrant=False)
        return fn(*args)

    @property
    def _fuse_checkpoints(self) -> bool:
        """checkpoint stem+backbone as ONE region when both halves are enabled.

        checkpointing them separately forces the boundary tensor -- the stem's 2D
        output, fp32 because groupnorm upcasts under autocast -- to be retained
        for every crop across the whole forward, since it is the input the second
        region needs to recompute from. at 256^2 that is 64*256*256*4 = 16.8 MB
        per global and 2.4 MB per local, so ~5 GB for a batch of 64 (globals +
        locals + the iBOT masked pass) that a single fused region never holds.
        fusing recomputes the stem inside the same region instead, which is what
        this encoder did before the two flags were split apart.
        """
        return self.training and self.checkpoint_stem and self.checkpoint_backbone

    def _fused(self, fn, x, *args):
        return checkpoint(fn, x, *args, use_reentrant=False)

    def _run(self, x: torch.Tensor) -> torch.Tensor:
        if self._fuse_checkpoints:
            return self._fused(self._forward_impl, x)
        return self._ckpt_backbone(self._backbone_pooled, self._ckpt_stem(x))

    def _run_tokens(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        if self._fuse_checkpoints:
            return self._fused(self._forward_tokens_impl, x, layer)
        return self._ckpt_backbone(self._backbone_tokens, self._ckpt_stem(x), layer)

    def _run_both(self, x: torch.Tensor, layer: str):
        if self._fuse_checkpoints:
            return self._fused(self._forward_both_impl, x, layer)
        return self._ckpt_backbone(
            self._backbone_pooled_and_tokens, self._ckpt_stem(x), layer
        )

    # fused stem+backbone bodies, used when both checkpoint flags are on.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone_pooled(self._stem_forward(x))

    def _forward_tokens_impl(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        return self._backbone_tokens(self._stem_forward(x), layer)

    def _forward_both_impl(self, x: torch.Tensor, layer: str):
        return self._backbone_pooled_and_tokens(self._stem_forward(x), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # split the crop batch so the stem never materializes all crops at once.
        # chunking runs in eval too (bounds the teacher's forward memory); only
        # the checkpointing inside _run is training-gated. concatenating chunk
        # outputs is exact: each crop is encoded independently.
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return torch.cat(
                [self._run(chunk) for chunk in x.split(self.chunk_size, dim=0)],
                dim=0,
            )
        return self._run(x)

    def forward_tokens(self, x, layer="layer4"):
        # chunk like forward() so the stem never materializes all crops at once
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return torch.cat(
                [self._run_tokens(chunk, layer) for chunk in x.split(self.chunk_size, dim=0)],
                dim=0,
            )
        return self._run_tokens(x, layer)

    def forward_with_tokens(self, x, layer="layer4"):
        """(pooled_features, layer tokens) from a single stem+backbone pass."""
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            pooled, tokens = [], []
            for chunk in x.split(self.chunk_size, dim=0):
                z, tok = self._run_both(chunk, layer)
                pooled.append(z)
                tokens.append(tok)
            return torch.cat(pooled, dim=0), torch.cat(tokens, dim=0)
        return self._run_both(x, layer)


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

    # forward_with_tokens must be exactly the two separate calls it replaces,
    # since the iBOT teacher now takes the fused path for both of its targets.
    small = HybridEncoder(stem_channels=16, latent_dim=64, chunk_size=2)
    small.eval()
    x = torch.randn(3, 1, 6, 64, 64)
    for layer in ("layer1", "layer2", "layer3", "layer4"):
        with torch.no_grad():
            z_ref = small(x)
            tok_ref = small.forward_tokens(x, layer=layer)
            z_fused, tok_fused = small.forward_with_tokens(x, layer=layer)
        assert torch.equal(z_ref, z_fused), f"pooled mismatch at {layer}"
        assert torch.equal(tok_ref, tok_fused), f"token mismatch at {layer}"
        print(f"  forward_with_tokens == (forward, forward_tokens) at {layer}  "
              f"pooled={tuple(z_fused.shape)} tokens={tuple(tok_fused.shape)}")

    # checkpointing is a memory/compute trade only: gradients must not depend on
    # which half (if either) is checkpointed.
    grads = {}
    for ckpt_stem, ckpt_bb in ((False, False), (True, False), (True, True)):
        torch.manual_seed(0)
        m = HybridEncoder(
            stem_channels=16, latent_dim=64, chunk_size=2,
            checkpoint_stem=ckpt_stem, checkpoint_backbone=ckpt_bb,
        )
        m.train()
        m(x).square().sum().backward()
        grads[(ckpt_stem, ckpt_bb)] = torch.cat(
            [p.grad.flatten() for p in m.parameters() if p.grad is not None]
        )
    base = grads[(False, False)]
    for key, g in grads.items():
        max_diff = (g - base).abs().max().item()
        assert max_diff < 1e-4, f"checkpointing changed gradients for {key}: {max_diff}"
        print(f"  checkpoint_stem={key[0]}, checkpoint_backbone={key[1]}: "
              f"max grad diff vs no checkpointing = {max_diff:.2e}")

    print("encoder smoke test passed.")
