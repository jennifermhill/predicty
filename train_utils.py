"""schedules, param groups, and checkpoint io for the WS-DINO training loop.

checkpoints always store the encoder under its own key, and also inside the
student. downstream eval can therefore load the encoder without the DINO
wrapper. see README "Conventions".
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

def linear_frac_schedule(
    step: int, total_steps: int, start: float, end: float, warmup_frac: float
) -> float:
    """linear warmup from `start` to `end` over `warmup_frac` of steps, then constant."""
    warmup_steps = max(1, int(total_steps * warmup_frac))
    if step < warmup_steps:
        return start + (end - start) * step / warmup_steps
    return end

def get_param_groups(model: torch.nn.Module, weight_decay: float) -> list[dict]:
    """split parameters into wd / no-wd groups.

    biases and 1-D parameters, that is norm scales, get no weight decay. the
    DINO recipe sets this rule, because a weight decay on them hurts.
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
        {"params": decay, "weight_decay": weight_decay, "wd_scheduled": True},
        {"params": no_decay, "weight_decay": 0.0, "wd_scheduled": False},
    ]


def batch_to_device(batch: Any, device: torch.device, non_blocking: bool = True) -> Any:
    """recursively move tensors in a nested dict/list batch to `device`."""
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: batch_to_device(v, device, non_blocking) for k, v in batch.items()}
    if isinstance(batch, (list, tuple)):
        return type(batch)(batch_to_device(v, device, non_blocking) for v in batch)
    return batch


def save_checkpoint(path, model, optimizer, scaler, step: int) -> None:
    """save full training state, plus the encoder under its own key."""
    state = {
        "encoder_state_dict": model.student.encoder.state_dict(),
        "student_state_dict": model.student.state_dict(),
        "teacher_state_dict": model.teacher.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "center": model.center.detach().cpu(),
        "ibot_center": model.ibot_center.detach().cpu() if hasattr(model, "ibot_center") else None,
        "step": int(step),
    }
    if getattr(model, "dataset_classifier", None) is not None:
        state["dataset_classifier_state_dict"] = model.dataset_classifier.state_dict()
    if getattr(model, "mae_decoder", None) is not None:
        state["mae_decoder_state_dict"] = model.mae_decoder.state_dict()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # write beside the destination and then replace atomically. latest.pt then
    # stays valid if something kills the job in the middle of a write.
    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        torch.save(state, tmp_path)
        os.replace(tmp_path, path)
    finally:
        tmp_path.unlink(missing_ok=True)


def load_checkpoint(
    path, model, optimizer=None, scaler=None
) -> int:
    """restore training state. the function returns the step of the checkpoint."""
    # weights_only: our checkpoints hold only tensors and primitive optimizer
    # state. the restricted loader therefore costs nothing, and it unpickles no
    # objects.
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
    if "ibot_center" in state and state["ibot_center"] is not None:
        if hasattr(model, "ibot_center"):
            model.ibot_center.copy_(state["ibot_center"].to(model.ibot_center.device))
    if optimizer is not None:
        try:
            optimizer.load_state_dict(state["optimizer_state_dict"])
        except ValueError as e:
            print(
                "[checkpoint] optimizer state is incompatible with the current "
                f"trainable modules; resetting optimizer ({e})"
            )
            # drop stale momentum buffers, do not carry them over
            for group in optimizer.param_groups:
                for p in group["params"]:
                    param_state = optimizer.state[p]
                    param_state.clear()
    if scaler is not None:
        # a bf16 run checkpoints an empty scaler state. a load of that state
        # into an fp16 run raises. start the scale fresh, because the scale
        # recalibrates in a few steps.
        scaler_state = state.get("scaler_state_dict") or {}
        if scaler_state:
            scaler.load_state_dict(scaler_state)
        elif scaler.is_enabled():
            print(
                "[checkpoint] no gradient scaler state in checkpoint (bf16 run?); "
                "starting the fp16 loss scale from its default"
            )
    return int(state["step"])
