"""Grad-CAM: visual explanations from deep networks via gradient-based localization.

the implementation follows Selvaraju et al. (2016):
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237336

Grad-CAM computes a class activation map. it weights each feature map by the
gradient of the class score.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradCAMHook:
    """the hook captures the activations and the gradients of a target layer."""

    def __init__(self, layer: nn.Module):
        self.gradients = None
        self.activations = None
        self.layer = layer

        self.fwd_hook = layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """the hook captures the activations in the forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """the hook captures the gradients in the backward pass."""
        self.gradients = grad_output[0]

    def remove(self):
        """the method removes the forward hook and the backward hook."""
        self.fwd_hook.remove()
        self.bwd_hook.remove()


class GradCAM:
    """Grad-CAM for any PyTorch model."""

    def __init__(
        self,
        encoder: nn.Module,
        classifier: nn.Module,
        target_layer: nn.Module,
        device: torch.device,
    ):
        """
        Args:
            encoder: feature extractor, for example HybridEncoder
            classifier: classification head, for example a linear probe
            target_layer: the layer that the hook attaches to, usually the
                final conv layer
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
        the method makes a Grad-CAM heatmap for one target class.

        Args:
            input_tensor: input image or crop, shape (1, C, Z, H, W) or (1, C, H, W)
            target_class: index of the target class for the explanation

        Returns:
            Grad-CAM heatmap, shape (H, W), with values in [0, 1]
        """
        self.encoder.eval()
        self.classifier.eval()

        self.hook = GradCAMHook(self.target_layer)

        input_tensor = input_tensor.to(self.device).requires_grad_(True)
        if input_tensor.size(0) != 1:
            raise ValueError(f"Expected batch size 1, got {input_tensor.size(0)}")
        with torch.enable_grad():
            embedding = self.encoder(input_tensor)
            logits = self.classifier(embedding)
            class_score = logits[0, target_class]

        self.encoder.zero_grad()
        self.classifier.zero_grad()
        class_score.backward()

        gradcam = self._compute_gradcam()

        self.hook.remove()
        input_tensor.requires_grad_(False)

        return gradcam

    def _compute_gradcam(self) -> torch.Tensor:
        """
        the method computes Grad-CAM from the captured gradients and activations.

        the paper gives three steps. first, a global average pool turns the
        gradients into per-channel weights. then a ReLU keeps the positive part
        of the weighted sum over the feature maps. last, the method normalizes
        the result to [0, 1].

        Returns:
            Grad-CAM heatmap (H, W)
        """
        gradients = self.hook.gradients      # (batch, C, H, W)
        activations = self.hook.activations  # (batch, C, H, W)

        alpha = gradients.mean(dim=(2, 3))  # (batch, C)

        # one op broadcasts the per-channel weights over the feature maps and
        # sums over the channels. a python loop is not necessary.
        gradcam = torch.relu(
            (alpha[0][:, None, None] * activations[0]).sum(dim=0)
        )

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
        the method makes a Grad-CAM for every class.

        Args:
            input_tensor: input image, shape (1, C, Z, H, W) or (1, C, H, W)
            num_classes: number of output classes

        Returns:
            a dict from class_idx to a Grad-CAM heatmap
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
    the function upsamples a Grad-CAM to the resolution of the input image.

    Args:
        gradcam: heatmap at the resolution of the feature map, shape (H_feat, W_feat)
        target_size: target spatial size (H_in, W_in)

    Returns:
        the upsampled heatmap, shape (H_in, W_in)
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
    the function extracts the final convolutional layer from a ResNet-based encoder.

    HybridEncoder is a 3D stem plus a 2D ResNet-50 backbone, so the last conv
    is layer4's final bottleneck block. for any other encoder, the function
    walks the module tree and returns the last Conv2d that it finds.
    """
    if hasattr(encoder, "backbone"):
        backbone = encoder.backbone
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1]

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
    """the function returns the 3D stem, the target for 3D attribution.

    ZAttentionPool collapses depth immediately after the stem. therefore the
    stem's stride-1 output is the last activation at full (Z, H, W) resolution.
    """
    if hasattr(encoder, "stem"):
        return encoder.stem
    raise ValueError(
        "encoder has no `stem`; 3D attribution needs a module whose output "
        "still has a z axis. specify target_layer explicitly."
    )


def _compute_layercam(
    activations: torch.Tensor, gradients: torch.Tensor
) -> torch.Tensor:
    """3D LayerCAM map from the encoder's 3D stem, before ZAttentionPool collapses z.

    a Grad-CAM on the 2D backbone attributes only in xy, at the coarse layer4
    resolution. that map has no z axis, because ZAttentionPool already collapsed
    z. the stem output is the last activation that is full-resolution and still
    (Z, H, W). the function therefore weights the stem output per voxel by its
    own positive gradient. global-average-pooled Grad-CAM instead weights the
    activation by a channel-averaged scalar. the per-voxel weight gives a true
    3D heatmap and hides no z information.

    `activations` and `gradients` are the (1, C, Z, H, W) stem tensors from a
    GradCAMHook. the function returns a (Z, H, W) map with values in [0, 1].
    """
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
    """the function normalizes to [0, 1] with the min and the max. a flat input
    maps to all zeros."""
    x = x - x.min()
    peak = x.max()
    return x / peak if peak > 1e-8 else torch.zeros_like(x)


class GradCAMAnalyzer:
    """all attributions for one crop from a single encoder forward pass.

    for each crop the analyzer returns the class probabilities and the predicted
    class. the analyzer also returns a 2D grad-cam for every class and
    ZAttentionPool's per-z depth weights. a 3D LayerCAM at the stem for one
    target class is optional.

    the 2D grad-cam needs no backward pass. layer4[-1] feeds straight into the
    backbone's global average pool, and the probe head is linear. the grad-cam
    channel weights for class c are therefore exactly that class's row in the
    weight matrix of the head, up to a positive constant that the normalization
    cancels. every class is a weighted sum of the same captured feature maps.
    only the 3D LayerCAM needs real gradients, at a cost of one backward pass
    for its single target class.
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
        self.final_conv = get_final_conv_layer(encoder)  # layer4[-1], before the pool
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
            n_classes: number of head outputs. the analyzer makes one 2D
                grad-cam for each output.
            cam3d_class: a class index. if cam_3d is enabled, the analyzer also
                makes a 3D LayerCAM for this class, at a cost of one backward
                pass.

        the method returns a dict with these keys: 'probs' (class_idx -> float),
        'pred_class' (int), 'cams_2d' (class_idx -> (h, w) tensor in [0, 1]),
        'z_weights' ((Z,) tensor whose entries sum to 1), and 'cam_3d'
        ((Z, H, W) tensor in [0, 1] or None).
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
        # the stem hook also captures gradients. the code attaches the stem hook
        # only for the 3D map.
        stem_hook = GradCAMHook(self.stem) if want_3d else None

        crop = crop.to(self.device)
        # only the 3D backward pass needs gradients. the analytic 2D maps, the
        # probabilities and the depth weights read from the forward pass alone.
        with torch.set_grad_enabled(want_3d):
            if want_3d:
                crop = crop.requires_grad_(True)
            embedding = self.encoder(crop)
            logits = self.classifier(embedding)

        prob_vec = torch.softmax(logits, dim=-1)[0]
        pred_class = int(prob_vec.argmax().item())
        probs = {i: float(prob_vec[i].item()) for i in range(n_classes)}

        # analytic 2D grad-cam for every class at once, with no backward pass:
        # map_c = relu(sum_k head.weight[c, k] * A_k). the code then normalizes
        # each class map.
        A = captured["conv"][0].detach()          # (C, h, w) pre-pool conv features
        W = self.classifier.weight.detach()       # (n_classes, latent_dim)
        # the encoder can project conv features (C) down to latent_dim before the
        # head. the code folds that projection into W, so that W lines up with
        # A's channels. the result is exact, because every step is linear.
        proj = getattr(self.encoder.backbone, "fc", None)
        if isinstance(proj, torch.nn.Linear):
            W = W @ proj.weight.detach()          # (n_classes, latent_dim) @ (latent_dim, C) -> (n_classes, C)
        maps = torch.einsum("nc,chw->nhw", W, A)
        cams_2d = {
            c: _normalize01(torch.relu(maps[c])).cpu() for c in range(n_classes)
        }

        # ZAttentionPool depth weights: the same reduction that the pool applies.
        score = captured["zscore"]                # (1, 1, Z, H, W)
        z_weights = score.mean(dim=(-1, -2)).softmax(dim=2)[0, 0].cpu()  # (Z,)

        # optional 3D LayerCAM: one backward pass for the single target class.
        cam_3d_map = None
        if want_3d:
            self.encoder.zero_grad()
            self.classifier.zero_grad()
            logits[0, cam3d_class].backward()
            cam_3d_map = _compute_layercam(
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
