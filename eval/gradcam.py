"""Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization.

Implementation based on Selvaraju et al. (2016):
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=8237336

Computes class activation maps by weighting feature maps by gradients of class score.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable


class GradCAMHook:
    """Hook to capture activations and gradients from a target layer."""

    def __init__(self, layer: nn.Module):
        self.gradients = None
        self.activations = None
        self.layer = layer

        # Register hooks
        self.fwd_hook = layer.register_forward_hook(self._forward_hook)
        self.bwd_hook = layer.register_full_backward_hook(self._backward_hook)

    def _forward_hook(self, module, input, output):
        """Capture activations during forward pass."""
        self.activations = output.detach()

    def _backward_hook(self, module, grad_input, grad_output):
        """Capture gradients during backward pass."""
        self.gradients = grad_output[0]

    def remove(self):
        """Clean up hooks."""
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
        Generate Grad-CAM heatmap for a target class.

        Args:
            input_tensor: Input image/crop, shape (1, C, Z, H, W) or (1, C, H, W)
            target_class: Index of target class for explanation

        Returns:
            Grad-CAM heatmap, shape (H, W) normalized to [0, 1]
        """
        self.encoder.eval()
        self.classifier.eval()

        # Setup hook
        self.hook = GradCAMHook(self.target_layer)

        # Forward pass
        input_tensor = input_tensor.to(self.device).requires_grad_(True)
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
        Compute Grad-CAM from hooked gradients and activations.

        Algorithm from paper:
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
        batch_size = gradients.size(0)
        num_channels = gradients.size(1)

        # Global average pool across spatial dimensions (2, 3)
        alpha = gradients.mean(dim=(2, 3))  # (batch, C)

        # Step 2: Weighted combination with ReLU
        # L^Grad-CAM = ReLU(Σ_k α_c^k * A^k)
        gradcam = torch.zeros(
            activations.shape[2:],
            device=activations.device,
            dtype=activations.dtype,
        )

        for c in range(num_channels):
            gradcam += alpha[0, c] * activations[0, c]

        gradcam = torch.relu(gradcam)

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
        Generate Grad-CAM for all classes.

        Args:
            input_tensor: Input image, shape (1, C, Z, H, W) or (1, C, H, W)
            num_classes: Number of output classes

        Returns:
            Dict mapping class_idx → Grad-CAM heatmap
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
    Upsample Grad-CAM to input image resolution.

    Args:
        gradcam: Heatmap from feature map resolution, shape (H_feat, W_feat)
        target_size: Target spatial size (H_in, W_in)

    Returns:
        Upsampled heatmap, shape (H_in, W_in)
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
    Extract the final convolutional layer from a ResNet-based encoder.

    For HybridEncoder with ResNet-50 body, returns the last conv block.

    Args:
        encoder: HybridEncoder or similar model

    Returns:
        Final conv layer module
    """
    # HybridEncoder structure: stem (3D) + body (2D ResNet-50)
    # ResNet-50 body: stem + layer1-4
    # Final conv layer is the last block of layer4

    if hasattr(encoder, "body"):
        # Assume encoder has a 'body' attribute (2D ResNet)
        body = encoder.body
        if hasattr(body, "layer4"):
            # ResNet has layer1, layer2, layer3, layer4
            layer4 = body.layer4
            # layer4[-1] is the last bottleneck block
            return layer4[-1]

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
