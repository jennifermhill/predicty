"""hybrid 3D-stem + 2D ResNet-50 encoder.

input  : (B, 1, Z, H, W)   Z is the slab depth (10 globals, 5 locals)
                            H = W is the spatial crop size (256 globals, 96 locals)
output : (B, latent_dim)    latent_dim is 2048 for ResNet-50

design:
  - a 3D stem runs first. then ZAttentionPool collapses z with learned
    per-slice weights. one set of weights handles both globals (Z=10) and
    locals (Z=5).
  - the 2D backbone is a ResNet-50. the first conv adapts to `stem_channels`,
    the classifier goes away, and the weights start random. ImageNet
    pretrained weights do not transfer to microscopy at the scale of this
    dataset.
  - warning: batchnorm is unsafe here. the ssl loop sends masked and
    multi-scale crops through this encoder. the degenerate statistics of those
    crops corrupt the running stats of batchnorm. the corrupt stats then ship
    in the frozen eval checkpoint. this encoder therefore uses groupnorm, which
    keeps no running-stat buffers.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint
from torchvision.models import resnet50


class ZAttentionPool(nn.Module):
    """learned z-aggregation. the module computes one softmaxed weight per
    z-slice, then the module makes a weighted sum. a fixed average-pool discards
    depth structure uniformly, so this module replaces the average-pool.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Conv3d(channels, 1, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, Z, H, W). the module averages over space for one score per
        # z-slice.
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

        # the dino wrapper flattens all crops into one batch. the
        # full-resolution 3D activations of the stem then dominate memory
        # (~20 GiB fp32 for 128 crops at 64ch/256x256x10). chunk_size caps the
        # crops per stem pass. None means no chunking.
        self.chunk_size = chunk_size

        # the two checkpointing halves have separate flags because their trade
        # is very different. the 3D full-res activations of the stem are the
        # memory hog. these activations are cheap to recompute. the backbone
        # holds much less memory, although the backbone holds most of the flops.
        # checkpointing of the backbone cost ~25-33% throughput. the default is
        # therefore stem only: enable checkpoint_backbone only on an oom.
        self.checkpoint_stem = bool(checkpoint_stem)
        self.checkpoint_backbone = bool(checkpoint_backbone)

        def make_norm(num_channels: int) -> nn.GroupNorm:
            # make_norm clamps groups to the largest divisor of num_channels
            # that is <= norm_groups. groupnorm then stays valid for any
            # channel count.
            groups = min(norm_groups, num_channels)
            while num_channels % groups != 0:
                groups -= 1
            return nn.GroupNorm(groups, num_channels)

        # the 3D stem keeps (Z, H, W) and grows the channel count. z_pool then
        # collapses z, so the 2D backbone sees one feature plane per sample.
        self.stem = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            make_norm(32),
            nn.GELU(),
            nn.Conv3d(32, stem_channels, kernel_size=3, padding=1, bias=False),
            make_norm(stem_channels),
            nn.GELU(),
        )
        self.z_pool = ZAttentionPool(stem_channels)

        # the code swaps the input conv (3 -> stem_channels) and drops the
        # classifier, because the encoder must output the 2048-d
        # pre-classifier feature. make_norm puts groupnorm everywhere.
        self.backbone = resnet50(weights=None, norm_layer=make_norm)
        self.backbone.conv1 = nn.Conv2d(
            stem_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        # nn.Identity at 2048 keeps the default checkpoint layout unchanged.
        # the final projection of resnet gives any other latent_dim.
        self.backbone.fc = (
            nn.Identity() if latent_dim == 2048 else nn.Linear(2048, latent_dim)
        )

    # ---- pieces of the forward, so each half can use checkpointing alone ----

    def _stem_forward(self, x: torch.Tensor) -> torch.Tensor:
        # (B, 1, Z, H, W) -> (B, stem_channels, H, W) for the 2D backbone
        x = self.stem(x)
        x = self.z_pool(x)
        return x.squeeze(2)

    def _backbone_pooled(self, x: torch.Tensor) -> torch.Tensor:
        # returns (B, latent_dim). fc is Identity when latent_dim == 2048.
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
        """one backbone pass that returns both the pooled feature and `layer`
        tokens.

        the iBOT teacher needs both targets from the same input. without this
        method the teacher runs the whole stem+backbone twice.
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
        if tok is None:  # "layer4", and any unrecognized value, as before
            tok = x
        z = torch.flatten(b.avgpool(x), 1)
        return b.fc(z), tok

    def _ckpt_stem(self, x: torch.Tensor) -> torch.Tensor:
        # groupnorm keeps its activations, unlike the fused cudnn batchnorm that
        # it replaced. the activations of the stem are 3D at full resolution.
        # checkpointing therefore earns its keep here. the code checkpoints in
        # training only, because inference has no backward to trade against.
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

        separate regions must keep the boundary tensor for every crop, because
        the second region recomputes from the boundary tensor. the boundary
        tensor is the 2D output of the stem, and it is fp32 because groupnorm
        upcasts under autocast. the cost is ~5 GB for a batch of 64. one fused
        region never holds the boundary tensor, because the fused region
        recomputes the stem inside itself.
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

    # fused stem+backbone bodies. the code uses them when both checkpoint flags
    # are on.
    def _forward_impl(self, x: torch.Tensor) -> torch.Tensor:
        return self._backbone_pooled(self._stem_forward(x))

    def _forward_tokens_impl(self, x: torch.Tensor, layer: str) -> torch.Tensor:
        return self._backbone_tokens(self._stem_forward(x), layer)

    def _forward_both_impl(self, x: torch.Tensor, layer: str):
        return self._backbone_pooled_and_tokens(self._stem_forward(x), layer)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward splits the crop batch, so the stem never materializes all
        # crops at once. chunking runs in eval too, because it bounds the
        # forward memory of the teacher. only the checkpointing inside _run is
        # training-gated. the result is exact either way, because the encoder
        # handles each crop independently.
        if self.chunk_size is not None and x.shape[0] > self.chunk_size:
            return torch.cat(
                [self._run(chunk) for chunk in x.split(self.chunk_size, dim=0)],
                dim=0,
            )
        return self._run(x)

    def forward_tokens(self, x, layer="layer4"):
        # forward_tokens chunks like forward(), so the stem never materializes
        # all crops at once
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
    # smoke test, no zarr needed. run as: python -m models.encoder
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

    # the iBOT teacher takes the fused path for both targets. the fused path
    # must therefore match the two separate calls that it replaces exactly.
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

    # checkpointing is a memory/compute trade only: gradients must not change.
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
