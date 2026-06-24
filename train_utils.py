"""utilities for the WS-DINO training loop.

schedules:
  lr_schedule           - linear warmup -> cosine decay to ~0
  ema_momentum_schedule - cosine from start to end over total_steps

param groups:
  get_param_groups - exclude biases and norm parameters from weight decay,
                     standard SSL practice. decaying them hurts.

device + checkpoints:
  batch_to_device   - recursive .to(device) for the dict batches our
                      DataLoader returns (tensors + a nested 'meta' dict).
  save_checkpoint   - dumps encoder + student + teacher + optimizer + scaler
                      + center + step. encoder is saved separately so it can
                      be loaded for downstream eval without the wrapper.
  load_checkpoint   - restores training state and returns the saved step.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch


def lr_schedule(
    step: int, total_steps: int, base_lr: float, warmup_frac: float
) -> float:
    """linear warmup over `warmup_frac * total_steps`, then cosine decay to 0."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return base_lr * step / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))


def ema_momentum_schedule(
    step: int, total_steps: int, start: float, end: float
) -> float:
    """cosine from `start` (at step 0) to `end` (at step total_steps)."""
    progress = min(step / max(1, total_steps), 1.0)
    return start + (end - start) * 0.5 * (1.0 - math.cos(math.pi * progress))


def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """split parameters into wd / no-wd groups.

    biases and 1-D parameters (norm scales) get no weight decay; everything
    else does. matches the DINO recipe.
    """
    decay: list[torch.nn.Parameter] = []
    no_decay: list[torch.nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim <= 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """recursively move tensors in a nested dict/list batch to `device`."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device, non_blocking) for v in batch)
    # ints, floats, strings, etc. pass through unchanged.
    return batch


def save_checkpoint(path, model, optimizer, scaler, step: int) -> None:
    """save full training state. encoder is also stored as its own key so it
    can be loaded for downstream eval without instantiating the DINO wrapper.
    """
    state = {
        "encoder_state_dict": model.student.encoder.state_dict(),
        "student_state_dict": model.student.state_dict(),
        "teacher_state_dict": model.teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "center": model.center.detach().cpu(),
        "step": int(step),
    }
    if getattr(model, "dataset_classifier", None) is not None:
        state["dataset_classifier_state_dict"] = model.dataset_classifier.state_dict()
    if getattr(model, "mae_decoder", None) is not None:
        state["mae_decoder_state_dict"] = model.mae_decoder.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write beside the destination and atomically replace it only after
    # serialization succeeds. This keeps latest.pt valid if a job is killed
    # during a checkpoint write.
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path, model, optimizer=None, scaler=None
) -> int:
    """restore training state. returns the step at which the checkpoint was saved."""
    # Checkpoints created here contain tensors and primitive optimizer state;
    # the restricted loader avoids executing arbitrary pickled objects.
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.student.load_state_dict(state["student_state_dict"])
    model.teacher.load_state_dict(state["teacher_state_dict"])
    if getattr(model, "dataset_classifier", None) is not None:
        if "dataset_classifier_state_dict" in state:
            model.dataset_classifier.load_state_dict(
                state["dataset_classifier_state_dict"]
            )
        else:
            print(
                "[checkpoint] no dataset classifier state in checkpoint; "
                "starting adversarial head from initialization"
            )
    if getattr(model, "mae_decoder", None) is not None:
        if "mae_decoder_state_dict" in state:
            model.mae_decoder.load_state_dict(state["mae_decoder_state_dict"])
        else:
            print(
                "[checkpoint] no MAE decoder state in checkpoint; "
                "starting MAE decoder from initialization"
            )
    model.center.copy_(state["center"].to(model.center.device))
    if optimizer is not None:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except ValueError as e:
            print(
                "[checkpoint] optimizer state is incompatible with the current "
                f"trainable modules; continuing with a fresh optimizer ({e})"
            )
    if scaler is not None:
        scaler.load_state_dict(state["scaler_state_dict"])
    return int(state["step"])
