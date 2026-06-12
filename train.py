"""WS-DINO training entry point.

usage:
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --resume checkpoints/latest.pt
    python train.py --config configs/train.yaml --device cpu   # smoke test

logs go to `cfg.log_dir` (tensorboard format). view with:
    tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset, worker_init_fn
from data.picker import pick_zarr_path
from models.dino import DINOWrapper, dino_loss
from train_utils import (
    batch_to_device,
    ema_momentum_schedule,
    get_param_groups,
    load_checkpoint,
    lr_schedule,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a yaml config")
    p.add_argument(
        "--resume", default=None, help="path to a checkpoint to resume from"
    )
    p.add_argument(
        "--device", default=None, help="override cfg.device (e.g. 'cpu', 'cuda:0')"
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="namespace checkpoints + logs under this run so concurrent runs "
        "don't clobber each other (checkpoints/<run>/, logs/<run>/)",
    )
    return p.parse_args()


def build_dataset(cfg: Config):
    if cfg.zarr_paths:
        return MultiZarrDataset(cfg)
    if cfg.zarr_path is None:
        cfg.zarr_path = pick_zarr_path()
    return ZarrCellDataset(cfg)


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    # namespace io under the run name so multiple concurrent runs keep their
    # checkpoints and tensorboard logs separate instead of overwriting.
    if args.run_name:
        cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / args.run_name)
        cfg.log_dir = str(Path(cfg.log_dir) / args.run_name)

    device = torch.device(cfg.device)
    # amp is only supported on cuda. silently disable on cpu.
    use_amp = cfg.use_amp and device.type == "cuda"

    print(f"[train] device={device}  use_amp={use_amp}")
    if device.type == "cuda":
        # confirms we actually got the gpu we asked the cluster for.
        idx = device.index if device.index is not None else 0
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        print(
            f"[train] cuda gpu name : {torch.cuda.get_device_name(idx)} "
            f"({total_gb:.1f} GB)"
        )
        print(f"[train] cuda device count: {torch.cuda.device_count()}")
    else:
        print("[train] WARNING: running on cpu — training will be impractically slow")
    print(f"[train] batch_size={cfg.batch_size}  total_steps={cfg.total_steps}")
    print(f"[train] base_lr={cfg.base_lr}  weight_decay={cfg.weight_decay}")

    # ---- data ----
    ds = build_dataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
    )

    # ---- model ----
    model = DINOWrapper(cfg).to(device)
    n_params = sum(p.numel() for p in model.student.parameters())
    print(f"[train] student params: {n_params:,}")
    # confirms .to(device) actually moved the model. if this prints 'cpu'
    # while device says 'cuda', training will silently run on cpu.
    print(f"[train] student params live on: {next(model.student.parameters()).device}")
    print(f"[train] teacher params live on: {next(model.teacher.parameters()).device}")

    # ---- optimizer + amp ----
    optimizer = AdamW(
        get_param_groups(model.student, cfg.weight_decay),
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler(device.type, enabled=use_amp)

    # ---- io ----
    # each invocation writes to its own timestamped subdirectory so multiple
    # runs stay separate in tensorboard's UI instead of accumulating into the
    # same log_dir as opaque event files. resumes get a `_resume` suffix so
    # they're visually distinguishable from fresh runs.
    run_name = time.strftime("%Y%m%d_%H%M%S")
    if args.resume:
        run_name = f"{run_name}_resume"
    run_log_dir = Path(cfg.log_dir) / run_name
    writer = SummaryWriter(run_log_dir)
    print(f"[train] tensorboard logs -> {run_log_dir}")
    ckpt_dir = Path(cfg.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # ---- resume ----
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optimizer, scaler) + 1
        print(f"[train] resumed from step {start_step}")

    # ---- training loop ----
    loader_iter = iter(loader)
    t_last_log = time.time()

    step = start_step
    for step in range(start_step, cfg.total_steps):
        # the IterableDataset is infinite, but guard against StopIteration anyway
        # so a worker restart doesn't kill training.
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch_to_device(batch, device)

        # on the first iteration only, confirm the batch actually landed on
        # the gpu. catches the "model is cuda but batches stay on cpu" bug
        # that silently triggers an implicit transfer per op.
        if step == start_step:
            print(f"[train] first batch globals device: {batch['globals'].device}")
            if "locals" in batch:
                print(f"[train] first batch locals  device: {batch['locals'].device}")

        # set LR per-step from the schedule.
        lr = lr_schedule(step, cfg.total_steps, cfg.base_lr, cfg.warmup_frac)
        for g in optimizer.param_groups:
            g["lr"] = lr

        # forward + loss under autocast (fp16 ops on cuda).
        with autocast(device.type, dtype=torch.float16, enabled=use_amp):
            out = model(batch)
            loss = dino_loss(
                out["student_globals"],
                out["student_locals"],
                out["teacher_globals"],
                model.center,
                out["B"], out["n_g"], out["n_l"],
                teacher_temp=cfg.teacher_temp,
                student_temp=cfg.student_temp,
            )

        # backward, unscale grads to apply clipping in fp32, step + update.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.student.parameters(), cfg.grad_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # EMA + center updates after the optimizer step.
        m = ema_momentum_schedule(
            step, cfg.total_steps, cfg.ema_momentum_start, cfg.ema_momentum_end
        )
        model.update_teacher(m)
        # center is updated on the raw teacher logits (float32 for stability).
        model.update_center(out["teacher_globals"].float(), cfg.center_momentum)

        # ---- logging ----
        if step % cfg.log_interval == 0:
            with torch.no_grad():
                # teacher entropy: high at init (uniform softmax), should
                # decrease then stabilize. crashing to 0 -> collapsed teacher.
                t_probs = (
                    (out["teacher_globals"] - model.center) / cfg.teacher_temp
                ).softmax(dim=-1)
                entropy = -(t_probs * t_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
                # embedding variance: stashed from the last student forward.
                # drops near 0 -> all crops mapping to the same point.
                # cast to fp32 because under autocast last_features is fp16,
                # and var of small fp16 values can underflow right when we
                # most need to see collapse coming.
                feats = model.student.last_features
                emb_var = (
                    feats.float().var(dim=0).mean()
                    if feats is not None
                    else torch.tensor(0.0)
                )
                center_norm = model.center.norm()

            dt = time.time() - t_last_log
            t_last_log = time.time()
            steps_per_sec = cfg.log_interval / max(dt, 1e-6) if step > start_step else 0.0

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("ema_momentum", m, step)
            writer.add_scalar("grad_norm", grad_norm.item(), step)
            writer.add_scalar("collapse/teacher_entropy", entropy.item(), step)
            writer.add_scalar("collapse/embedding_var", emb_var.item(), step)
            writer.add_scalar("collapse/center_norm", center_norm.item(), step)
            writer.add_scalar("throughput/steps_per_sec", steps_per_sec, step)

            print(
                f"step {step:06d}  loss={loss.item():.4f}  lr={lr:.2e}  "
                f"ent={entropy.item():.3f}  emb_var={emb_var.item():.4f}  "
                f"center_norm={center_norm.item():.3f}  {steps_per_sec:.2f} step/s"
            )

        # ---- checkpoint ----
        if step > 0 and step % cfg.ckpt_interval == 0:
            save_checkpoint(
                ckpt_dir / f"step_{step:06d}.pt", model, optimizer, scaler, step
            )
            save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler, step)

    # final checkpoint after the last step.
    save_checkpoint(ckpt_dir / "final.pt", model, optimizer, scaler, step)
    save_checkpoint(ckpt_dir / "latest.pt", model, optimizer, scaler, step)
    writer.close()
    print("[train] training complete.")


if __name__ == "__main__":
    main()
