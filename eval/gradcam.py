"""Grad-CAM: visual explanations from deep networks via gradient-based localization.

implementation based on Selvaraju et al. (2016):
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237336

computes class activation maps by weighting feature maps by gradients of the class score.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMHook:
    """hook to capture activations and gradients from a target layer."""

    def __init__(self, layer: nn.Module):
        self.gradients = None
        self.activations = None
        self.layer = layer

        # Register hooks
        self.fwd_hook = layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """capture activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """capture gradients during backward pass."""
        self.gradients = grad_output[0]

    def remove(self):
        """clean up hooks."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()


class GradCAM:
    """Grad-CAM implementation for any PyTorch model."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            encoder: Feature extractor (e.g., HybridEncoder)
            classifier: Classification head (e.g., Linear probe)
            target_layer: Layer to hook (usually final conv layer)
            device: torch device
        """
        self.encoder = encoder
        self.classifier = classifier
        self.target_layer = target_layer
        self.device = device
        self.hook = None

    def generate_gradcam(
        self,
        input_tensor: torch.Tensor,
        target_class: int,
    ) -> torch.Tensor:
        """
        generate Grad-CAM heatmap for a target class.

        Args:
            input_tensor: input image/crop, shape (1, C, Z, H, W) or (1, C, H, W)
            target_class: index of target class for explanation

        Returns:
            Grad-CAM heatmap, shape (H, W) normalized to [0, 1]
        """
        self.encoder.eval()
        self.classifier.eval()

        # Setup hook
        self.hook = GradCAMHook(self.target_layer)

        # Forward pass
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        if input_tensor.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got {input_tensor.size(0)}")
        with torch.enable_grad():
            embedding = self.encoder(input_tensor)
            logits = self.classifier(embedding)
            class_score = logits[0, target_class]

        # Backward pass
        self.encoder.zero_grad()
        self.classifier.zero_grad()
        class_score.backward()

        # Grad-CAM algorithm
        gradcam = self._compute_gradcam()

        # Cleanup
        self.hook.remove()
        input_tensor.requires_grad_(False)

        return gradcam

    def _compute_gradcam(self) -> torch.Tensor:
        """
        compute Grad-CAM from hooked gradients and activations.

        algorithm from paper:
        1. Compute channel weights: α_c^k = (1/Z) * Σ_i Σ_j (∂y^c / ∂A_{ij}^k)
        2. Weighted combination: L^Grad-CAM = ReLU(Σ_k α_c^k * A^k)
        3. Normalize to [0, 1]

        Returns:
            Grad-CAM heatmap (H, W)
        """
        # Extract gradients and activations
        # gradients: (batch, C, H, W)
        # activations: (batch, C, H, W)
        gradients = self.hook.gradients
        activations = self.hook.activations

        # Step 1: Compute channel weights by global average pooling gradients
        # α_c^k = (1/Z) * Σ_i Σ_j (∂y^c / ∂A_{ij}^k)
        # Global average pool across spatial dimensions (2, 3)
        alpha = gradients.mean(dim=(2, 3))  # (batch, C)

        # Step 2: Weighted combination with ReLU
        # L^Grad-CAM = ReLU(Σ_k α_c^k * A^k). broadcast the per-channel weights
        # over the feature maps and sum over channels in one op (no python loop).
        gradcam = torch.relu(
            (alpha[0][:, None, None] * activations[0]).sum(dim=0)
        )

        # Step 3: Normalize to [0, 1]
        gradcam_min = gradcam.min()
        gradcam_max = gradcam.max()

        if gradcam_max - gradcam_min > 1e-8:
            gradcam = (gradcam - gradcam_min) / (gradcam_max - gradcam_min)
        else:
            gradcam = torch.zeros_like(gradcam)

        return gradcam.cpu()

    def generate_gradcam_multiclass(
        self,
        input_tensor: torch.Tensor,
        num_classes: int,
    ) -> dict[int, torch.Tensor]:
        """
        generate Grad-CAM for all classes.

        Args:
            input_tensor: input image, shape (1, C, Z, H, W) or (1, C, H, W)
            num_classes: number of output classes

        Returns:
            dict mapping class_idx -> Grad-CAM heatmap
        """
        results = {}
        for class_idx in range(num_classes):
            gradcam = self.generate_gradcam(input_tensor, class_idx)
            results[class_idx] = gradcam
        return results


def upsample_gradcam(
    gradcam: torch.Tensor,
    target_size: tuple[int, int],
) -> torch.Tensor:
    """
    upsample Grad-CAM to input image resolution.

    Args:
        gradcam: heatmap from feature map resolution, shape (H_feat, W_feat)
        target_size: target spatial size (H_in, W_in)

    Returns:
        upsampled heatmap, shape (H_in, W_in)
    """
    if gradcam.dim() == 2:
        gradcam = gradcam.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    upsampled = F.interpolate(
        gradcam,
        size=target_size,
        mode="bilinear",
        align_corners=False,
    )

    return upsampled.squeeze()  # (H, W)


def get_final_conv_layer(encoder: nn.Module) -> nn.Module:
    """
    extract the final convolutional layer from a ResNet-based encoder.

    for HybridEncoder with a ResNet-50 backbone, returns the last conv block.

    Args:
        encoder: HybridEncoder or similar model

    Returns:
        final conv layer module
    """
    # HybridEncoder structure: stem (3D) + backbone (2D ResNet-50)
    # ResNet-50 backbone: conv stem + layer1-4
    # Final conv layer is the last block of layer4

    if hasattr(encoder, "backbone"):
        backbone = encoder.backbone
        if hasattr(backbone, "layer4"):
            # ResNet has layer1, layer2, layer3, layer4
            # layer4[-1] is the last bottleneck block
            return backbone.layer4[-1]

    # Fallback: recursively find last Conv2d
    last_conv = None
    for module in encoder.modules():
        if isinstance(module, nn.Conv2d):
            last_conv = module

    if last_conv is None:
        raise ValueError(
            "Could not find final conv layer. Specify target_layer explicitly."
        )

    return last_conv


def get_stem_layer(encoder: nn.Module) -> nn.Module:
    """return the 3D stem, whose output is the last full-resolution 3D tensor.

    HybridEncoder collapses the depth axis in ZAttentionPool right after the
    stem, so the stem output (B, stem_channels, Z, H, W) is the last activation
    that still has a z axis. it is also stride-1, so it is at full input
    (Z, H, W) resolution. this is the target for 3D attribution.
    """
    if hasattr(encoder, "stem"):
        return encoder.stem
    raise ValueError(
        "encoder has no `stem`; 3D attribution needs a module whose output "
        "still has a z axis. specify target_layer explicitly."
    )


@torch.no_grad()
def zattention_weights(
    encoder: nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """ZAttentionPool's per-z-slice softmax weights for a single crop.

    these are the model's literal depth weighting: one number per z-slice,
    spatially uniform (the pool averages its score over H, W before the softmax
    over z). that makes this the honest companion to LayerCAM3D's per-voxel map,
    which can show spatial-per-z variation the pool never actually uses to
    weight depth. re-derives the weights exactly as ZAttentionPool.forward does
    by hooking its score conv. returns a (Z,) tensor that sums to 1.
    """
    if not hasattr(encoder, "z_pool") or not hasattr(encoder.z_pool, "score"):
        raise ValueError("encoder has no ZAttentionPool `z_pool.score` to read")
    if input_tensor.size(0) != 1:
        raise ValueError(f"expected batch size 1, got {input_tensor.size(0)}")

    captured: dict[str, torch.Tensor] = {}

    def hook(module, inp, out):
        captured["score"] = out.detach()  # (1, 1, Z, H, W)

    handle = encoder.z_pool.score.register_forward_hook(hook)
    encoder.eval()
    try:
        encoder(input_tensor.to(device))
    finally:
        handle.remove()

    score = captured["score"]  # (1, 1, Z, H, W)
    # same reduction as the pool: spatial mean per slice, then softmax over z.
    weights = score.mean(dim=(-1, -2)).softmax(dim=2)  # (1, 1, Z)
    return weights[0, 0].detach().cpu()  # (Z,)


class LayerCAM3D:
    """3D LayerCAM at the encoder's 3D stem, before the depth axis is collapsed.

    a Grad-CAM on the 2D backbone can only attribute in xy at the coarse layer4
    resolution, and has no z axis at all (ZAttentionPool already collapsed it).
    this hooks the stem output instead, the last activation that is still
    (Z, H, W) and full-resolution, and forms a per-voxel map with LayerCAM's
    element-wise positive-gradient weighting (better than global-average-pooled
    Grad-CAM weights at a shallow layer). the result is a genuine 3D heatmap the
    same shape as the input crop, so no z information is hidden.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
        target_layer: Optional[nn.Module] = None,
    ) -> None:
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        # default to the 3D stem output (the z_pool input).
        self.target_layer = target_layer if target_layer is not None else get_stem_layer(encoder)

    def generate(self, input_tensor: torch.Tensor, target_class: int) -> torch.Tensor:
        """return a (Z, H, W) attribution map in [0, 1] for `target_class`.

        Args:
            input_tensor: single crop, shape (1, 1, Z, H, W).
            target_class: class index whose score is attributed.
        """
        if input_tensor.size(0) != 1:
            raise ValueError(f"expected batch size 1, got {input_tensor.size(0)}")

        self.encoder.eval()
        self.classifier.eval()
        hook = GradCAMHook(self.target_layer)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        with torch.enable_grad():
            embedding = self.encoder(input_tensor)
            logits = self.classifier(embedding)
            class_score = logits[0, target_class]

        self.encoder.zero_grad()
        self.classifier.zero_grad()
        class_score.backward()

        cam = self._compute_layercam(hook.activations, hook.gradients)

        hook.remove()
        input_tensor.requires_grad_(False)
        return cam

    @staticmethod
    def _compute_layercam(
        activations: torch.Tensor, gradients: torch.Tensor
    ) -> torch.Tensor:
        # activations, gradients: (1, C, Z, H, W). LayerCAM weights each voxel by
        # its own positive gradient (not a channel-averaged scalar), which keeps
        # spatial + depth structure that global-average-pool grad-cam would blur.
        weights = torch.relu(gradients)
        cam = (weights * activations).sum(dim=1)  # (1, Z, H, W)
        cam = torch.relu(cam)[0]  # (Z, H, W)
        cam = cam - cam.min()
        peak = cam.max()
        if peak > 1e-8:
            cam = cam / peak
        else:
            cam = torch.zeros_like(cam)
        return cam.detach().cpu()


def _normalize01(x: torch.Tensor) -> torch.Tensor:
    """min-max normalize to [0, 1]; a flat input maps to all zeros."""
    x = x - x.min()
    peak = x.max()
    return x / peak if peak > 1e-8 else torch.zeros_like(x)


class GradCAMAnalyzer:
    """all attributions for one crop from a single encoder forward.

    returns, per crop: class probabilities + predicted class, a 2D grad-cam for
    every class, ZAttentionPool's per-z depth weights, and (optionally) a 3D
    LayerCAM at the stem for one target class.

    the 2D grad-cam needs no backward pass. the target layer (layer4[-1]) feeds
    straight into the backbone's global-average-pool and the probe head is
    linear, so grad-cam's channel weights for class c are exactly that class's
    head weight row (up to a positive constant that cancels in normalization).
    every class is then a weighted sum of the same captured feature maps. only
    the 3D LayerCAM needs real gradients (the stem sits deep in the net), which
    costs one backward on the retained graph for its single target class.

    net cost per crop: one forward, plus one backward only when a 3D map is
    requested, versus the previous (n_classes + 3) forwards and (n_classes + 1)
    backwards.
    """

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        device: torch.device,
        cam_3d: bool = False,
    ) -> None:
        if not isinstance(classifier, nn.Linear):
            raise TypeError(
                "GradCAMAnalyzer's analytic 2D path assumes a linear probe head; "
                f"got {type(classifier).__name__}. use the GradCAM class (per-class "
                "backprop) for a non-linear head."
            )
        if not hasattr(encoder, "z_pool") or not hasattr(encoder.z_pool, "score"):
            raise ValueError("encoder has no ZAttentionPool `z_pool.score` to read")
        self.encoder = encoder
        self.classifier = classifier
        self.device = device
        self.cam_3d = cam_3d
        self.final_conv = get_final_conv_layer(encoder)  # layer4[-1], pre-GAP
        self.stem = get_stem_layer(encoder) if cam_3d else None
        self.z_score = encoder.z_pool.score

    def analyze(
        self,
        crop: torch.Tensor,
        n_classes: int,
        cam3d_class: Optional[int] = None,
    ) -> dict:
        """
        Args:
            crop: single crop, shape (1, 1, Z, H, W).
            n_classes: number of head outputs (2D grad-cam is made for each).
            cam3d_class: if given and cam_3d is enabled, also make a 3D LayerCAM
                for this class (one backward pass).

        Returns dict with keys: 'probs' (class_idx -> float), 'pred_class' (int),
        'cams_2d' (class_idx -> (h, w) tensor in [0, 1]), 'z_weights' ((Z,) tensor
        summing to 1), and 'cam_3d' ((Z, H, W) tensor in [0, 1] or None).
        """
        if crop.size(0) != 1:
            raise ValueError(f"expected batch size 1, got {crop.size(0)}")
        self.encoder.eval()
        self.classifier.eval()

        captured: dict[str, torch.Tensor] = {}
        handles = [
            self.final_conv.register_forward_hook(
                lambda m, i, o: captured.__setitem__("conv", o)
            ),
            self.z_score.register_forward_hook(
                lambda m, i, o: captured.__setitem__("zscore", o.detach())
            ),
        ]
        want_3d = self.cam_3d and cam3d_class is not None
        # the stem hook also grabs gradients, so only attach it when doing 3D.
        stem_hook = GradCAMHook(self.stem) if want_3d else None

        crop = crop.to(self.device)
        # gradients are needed only for the 3D backward; the analytic 2D maps,
        # probabilities, and depth weights all read from the forward alone.
        with torch.set_grad_enabled(want_3d):
            if want_3d:
                crop = crop.requires_grad_(True)
            embedding = self.encoder(crop)
            logits = self.classifier(embedding)

        # probabilities + prediction, straight from the shared forward.
        prob_vec = torch.softmax(logits, dim=-1)[0]
        pred_class = int(prob_vec.argmax().item())
        probs = {i: float(prob_vec[i].item()) for i in range(n_classes)}

        # analytic 2D grad-cam for every class at once (no backward):
        # map_c = relu(sum_k head.weight[c, k] * A_k), then per-class normalize.
        A = captured["conv"][0].detach()          # (C, h, w) pre-pool conv features
        W = self.classifier.weight.detach()       # (n_classes, latent_dim)
        # if the encoder projects conv features (C) down to latent_dim before the head,
        # fold that projection in so W lines up with A's channels. exact: it's all linear.
        proj = getattr(self.encoder.backbone, "fc", None)
        if isinstance(proj, torch.nn.Linear):
            W = W @ proj.weight.detach()          # (n_classes, latent_dim) @ (latent_dim, C) -> (n_classes, C)
        maps = torch.einsum("nc,chw->nhw", W, A)
        cams_2d = {
            c: _normalize01(torch.relu(maps[c])).cpu() for c in range(n_classes)
        }

        # ZAttentionPool depth weights: same reduction the pool applies.
        score = captured["zscore"]                # (1, 1, Z, H, W)
        z_weights = score.mean(dim=(-1, -2)).softmax(dim=2)[0, 0].cpu()  # (Z,)

        # optional 3D LayerCAM: one backward for the single target class.
        cam_3d_map = None
        if want_3d:
            self.encoder.zero_grad()
            self.classifier.zero_grad()
            logits[0, cam3d_class].backward()
            cam_3d_map = LayerCAM3D._compute_layercam(
                stem_hook.activations, stem_hook.gradients
            )
            stem_hook.remove()
            crop.requires_grad_(False)

        for h in handles:
            h.remove()

        return {
            "probs": probs,
            "pred_class": pred_class,
            "cams_2d": cams_2d,
            "z_weights": z_weights,
            "cam_3d": cam_3d_map,
        }
