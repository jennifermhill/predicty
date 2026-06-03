"""DINO projector, student/teacher wrapper, and loss.

projector ("DINO head"):
  encoder output (B, latent_dim) -> 3-layer MLP -> L2-normalized bottleneck
  -> weight-normalized linear -> K prototype logits

wrapper:
  student = encoder + projector (trainable)
  teacher = EMA copy of student (no grad)
  center  = running mean of teacher logits; subtracted before softmax to
            prevent any one prototype from dominating (DINO's main anti-collapse
            mechanism).

loss: cross-entropy between (centered, sharpened) teacher and (sharpened)
student over all (teacher_global, student_view) pairs except same-view
global-global. with 2 globals + n_local locals there are
2 * (n_global - 1) + 2 * n_local = 2 + 2 * n_local pair terms per sample.
"""

from __future__ import annotations

import copy
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class DINOHead(nn.Module):
    """3-layer MLP -> L2 normalize -> weight-normed prototype layer.

    the prototype layer's magnitude (`parametrizations.weight.original0`) is
    frozen at 1 — the "norm_last_layer" trick from DINO. removes a degree of
    freedom that otherwise lets the model cheat on softmax confidence by
    rescaling.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 2048,
        bottleneck_dim: int = 256,
        out_dim: int = 4096,
        norm_last_layer: bool = True,
    ) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        # use the parametrizations API (not legacy `nn.utils.weight_norm`):
        # the legacy version registers `weight` as a non-leaf tensor, which
        # breaks `copy.deepcopy` and therefore the teacher-from-student init
        # below. with parametrizations the magnitude lives at
        # `last_layer.parametrizations.weight.original0` and is a real leaf,
        # so deepcopy works and the freeze trick is still available.
        self.last_layer = nn.utils.parametrizations.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.parametrizations.weight.original0.data.fill_(1)
        if norm_last_layer:
            self.last_layer.parametrizations.weight.original0.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.mlp(x)
        z = F.normalize(z, dim=-1, p=2)
        return self.last_layer(z)


class StudentNet(nn.Module):
    """encoder + projector. stashes the last encoder features (no grad) so the
    training loop can compute embedding variance without a redundant forward."""

    def __init__(self, encoder: nn.Module, projector: nn.Module) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = projector
        self.last_features: Optional[torch.Tensor] = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        self.last_features = z.detach()
        return self.projector(z)


class DINOWrapper(nn.Module):
    """owns the trainable student, the EMA teacher, and the centering buffer."""

    def __init__(self, cfg) -> None:
        super().__init__()
        # local import keeps the encoder module decoupled from the wrapper.
        from models.encoder import HybridEncoder

        student_enc = HybridEncoder(stem_channels=cfg.stem_channels)
        student_proj = DINOHead(
            in_dim=cfg.latent_dim,
            hidden_dim=cfg.projector_hidden,
            bottleneck_dim=cfg.projector_bottleneck,
            out_dim=cfg.projector_k,
        )
        self.student = StudentNet(student_enc, student_proj)

        # teacher: frozen deep-copy of the student. EMA-updated each step.
        self.teacher = copy.deepcopy(self.student)
        for p in self.teacher.parameters():
            p.requires_grad = False

        # centering buffer: EMA of pre-softmax teacher logits. registered as a
        # buffer so it moves with `.to(device)` and is saved in checkpoints.
        self.register_buffer("center", torch.zeros(cfg.projector_k))

    def forward(self, batch: dict) -> dict:
        globals_ = batch["globals"]  # (B, 2, 1, Z, H, W)
        locals_ = batch.get("locals")  # (B, n_l, 1, Z, H, W) or None
        B = globals_.shape[0]
        n_g = globals_.shape[1]

        # student processes globals + locals; teacher only globals.
        g_flat = globals_.flatten(0, 1)
        s_g = self.student(g_flat)

        if locals_ is not None and locals_.numel() > 0:
            n_l = locals_.shape[1]
            l_flat = locals_.flatten(0, 1)
            s_l = self.student(l_flat)
        else:
            n_l = 0
            s_l = None

        with torch.no_grad():
            t_g = self.teacher(g_flat)

        return {
            "student_globals": s_g,
            "student_locals": s_l,
            "teacher_globals": t_g,
            "B": B,
            "n_g": n_g,
            "n_l": n_l,
        }

    @torch.no_grad()
    def update_teacher(self, momentum: float) -> None:
        """teacher = m * teacher + (1 - m) * student, parameter-wise."""
        for p_s, p_t in zip(self.student.parameters(), self.teacher.parameters()):
            p_t.data.mul_(momentum).add_(p_s.data, alpha=1.0 - momentum)

    @torch.no_grad()
    def update_center(
        self, teacher_logits: torch.Tensor, momentum: float = 0.9
    ) -> None:
        """EMA the centering buffer toward this batch's teacher mean."""
        self.center.mul_(momentum).add_(
            teacher_logits.mean(dim=0), alpha=1.0 - momentum
        )


def dino_loss(
    student_globals: torch.Tensor,
    student_locals: Optional[torch.Tensor],
    teacher_globals: torch.Tensor,
    center: torch.Tensor,
    B: int,
    n_g: int,
    n_l: int,
    teacher_temp: float = 0.04,
    student_temp: float = 0.1,
) -> torch.Tensor:
    """cross-entropy between centered+sharp teacher and sharp student logits."""
    # teacher: center, sharpen, softmax — detach (no grad through teacher).
    t = ((teacher_globals - center) / teacher_temp).softmax(dim=-1).detach()
    t = t.view(B, n_g, -1)

    # student: sharpen, log-softmax
    sg = (student_globals / student_temp).log_softmax(dim=-1).view(B, n_g, -1)
    sl = (
        (student_locals / student_temp).log_softmax(dim=-1).view(B, n_l, -1)
        if student_locals is not None and n_l > 0
        else None
    )

    total = student_globals.new_zeros(())
    n_terms = 0

    # global-global: skip same-view pairs (the identity match is trivial).
    for ti in range(n_g):
        for sj in range(n_g):
            if ti == sj:
                continue
            total = total + -(t[:, ti] * sg[:, sj]).sum(dim=-1).mean()
            n_terms += 1

    # local-global: every (teacher global, student local) pair.
    if sl is not None:
        for ti in range(n_g):
            for sj in range(n_l):
                total = total + -(t[:, ti] * sl[:, sj]).sum(dim=-1).mean()
                n_terms += 1

    return total / max(n_terms, 1)


if __name__ == "__main__":
    # smoke test with random tensors. no zarr required.
    # run from project root: python -m models.dino
    from configs.config import Config

    cfg = Config()
    cfg.projector_k = 256  # tiny K for fast test

    wrapper = DINOWrapper(cfg)
    wrapper.eval()

    B = 2
    batch = {
        # smaller spatial dims than real config for a quick CPU smoke test.
        "globals": torch.randn(B, 2, 1, 10, 96, 96),
        "locals": torch.randn(B, 6, 1, 5, 48, 48),
    }
    out = wrapper(batch)
    print(
        f"student_globals: {tuple(out['student_globals'].shape)}  "
        f"(expected ({B * 2}, {cfg.projector_k}))"
    )
    print(
        f"student_locals : {tuple(out['student_locals'].shape)}   "
        f"(expected ({B * 6}, {cfg.projector_k}))"
    )
    print(
        f"teacher_globals: {tuple(out['teacher_globals'].shape)}  "
        f"(expected ({B * 2}, {cfg.projector_k}))"
    )

    loss = dino_loss(
        out["student_globals"],
        out["student_locals"],
        out["teacher_globals"],
        wrapper.center,
        out["B"], out["n_g"], out["n_l"],
        cfg.teacher_temp, cfg.student_temp,
    )
    print(f"loss: {loss.item():.4f}")
    assert torch.isfinite(loss), "loss is not finite"

    loss.backward()
    n_with_grad = sum(
        1
        for p in wrapper.student.parameters()
        if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_with_grad > 0, "no gradient flowed to the student"
    print(f"backward: {n_with_grad} student parameters got gradient")
    print("dino.py smoke test passed.")
