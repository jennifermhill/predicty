"""WS-DINO training entry point.

usage:
    python train.py --config configs/train.yaml
    python train.py --config configs/train.yaml --resume checkpoints/latest.pt
    python train.py --config configs/train.yaml --device cpu   # smoke test

logs go to `cfg.log_dir` in tensorboard format. view the logs with:
    tensorboard --logdir logs
"""

from __future__ import annotations

import argparse
import atexit
import shutil
import socket
import subprocess
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset, worker_init_fn
from data.picker import pick_zarr_path
from models.dino import (
    DINOWrapper,
    dino_loss,
    ibot_patch_loss,
    ibot_patch_accuracy,
)
from train_utils import (
    batch_to_device,
    ema_momentum_schedule,
    get_param_groups,
    load_checkpoint,
    lr_schedule,
    linear_frac_schedule,
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


def find_free_port(start: int, attempts: int = 50) -> int | None:
    """first free TCP port at or above `start`, or None if the range is full.

    each concurrent run on one node can then get its own tensorboard.
    """
    for port in range(start, start + attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            # with SO_REUSEADDR a socket in TIME_WAIT does not look taken.
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            try:
                s.bind(("", port))
                return port
            except OSError:
                continue
    return None


def launch_tensorboard(log_dir: str, port: int) -> subprocess.Popen | None:
    """start a tensorboard server on log_dir and tear it down at exit.

    the server is a child process, so a missing binary or a busy port can never
    take down training. the function returns the handle, or None if the server
    cannot start.
    """
    tb_bin = shutil.which("tensorboard")
    if tb_bin is None:
        print("[train] tensorboard not found on PATH, skipping auto-launch")
        return None

    free_port = find_free_port(port)
    if free_port is None:
        print(
            f"[train] no free port in [{port}, {port + 50}), skipping tensorboard"
        )
        return None
    if free_port != port:
        print(f"[train] port {port} busy, tensorboard using {free_port} instead")

    Path(log_dir).mkdir(parents=True, exist_ok=True)
    try:
        proc = subprocess.Popen(
            [tb_bin, "--logdir", log_dir, "--port", str(free_port), "--bind_all"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except OSError as e:
        print(f"[train] failed to launch tensorboard: {e}")
        return None

    # the server must not outlive the job and hold the port.
    atexit.register(lambda: proc.terminate())
    host = socket.gethostname()
    print(
        f"[train] tensorboard serving {log_dir} at "
        f"http://{host}:{free_port} (pid {proc.pid})"
    )
    return proc


def main() -> None:
    args = parse_args()
    cfg = Config.from_yaml(args.config)
    if args.device:
        cfg.device = args.device
    # main captures the root log dir before it namespaces the run. tensorboard
    # then serves every run, like the manual `tensorboard --logdir logs`.
    tensorboard_root = cfg.log_dir
    # namespace io per run, so concurrent runs do not overwrite each other.
    if args.run_name:
        cfg.checkpoint_dir = str(Path(cfg.checkpoint_dir) / args.run_name)
        cfg.log_dir = str(Path(cfg.log_dir) / args.run_name)

    if cfg.tensorboard_enabled:
        launch_tensorboard(tensorboard_root, cfg.tensorboard_port)

    device = torch.device(cfg.device)
    # only cuda supports amp. the code disables amp on cpu without a message.
    use_amp = cfg.use_amp and device.type == "cuda"

    # warning: bf16 is the wrong choice on this data. the 8 mantissa bits of
    # bf16 collapse the teacher. see README "Results: where the models stand".
    # bf16 carries the exponent range of fp32, so bf16 needs no gradient
    # scaling. the code falls back to fp16 where the hardware lacks bf16.
    amp_dtype_name = str(getattr(cfg, "amp_dtype", "fp16")).lower()
    if amp_dtype_name not in ("bf16", "bfloat16", "fp16", "float16"):
        raise ValueError(
            f"cfg.amp_dtype must be 'bf16' or 'fp16'; got {cfg.amp_dtype!r}"
        )
    want_bf16 = amp_dtype_name in ("bf16", "bfloat16")
    if want_bf16 and use_amp and not torch.cuda.is_bf16_supported():
        print("[train] WARNING: bf16 unsupported on this gpu, falling back to fp16")
        want_bf16 = False
    amp_dtype = torch.bfloat16 if want_bf16 else torch.float16
    # the scaler is fp16-only. under bf16 every scaler call is a passthrough.
    use_scaler = use_amp and amp_dtype is torch.float16

    print(
        f"[train] device={device}  use_amp={use_amp}  "
        f"amp_dtype={'bf16' if want_bf16 else 'fp16'}  grad_scaler={use_scaler}"
    )
    if device.type == "cuda":
        # the print below confirms the gpu that the cluster gave us.
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
    print(f"[train] base_lr={cfg.base_lr}  weight_decay={cfg.weight_decay_start} -> {cfg.weight_decay_end}")

    # ---- data ----
    ds = build_dataset(cfg)
    n_datasets = len(ds.children) if isinstance(ds, MultiZarrDataset) else 1
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
        pin_memory=device.type == "cuda",
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )

    # ---- model ----
    model = DINOWrapper(cfg, n_datasets=n_datasets).to(device)
    n_params = sum(p.numel() for p in model.student.parameters())
    print(f"[train] student params: {n_params:,}")
    if model.dataset_classifier is not None:
        n_adv_params = sum(p.numel() for p in model.dataset_classifier.parameters())
        print(
            f"[train] dataset adversary: enabled "
            f"(classes={n_datasets}, params={n_adv_params:,}, "
            f"weight={cfg.dataset_adversary_weight}, "
            f"grl_scale={model.dataset_adversary_grl_scale})"
        )
    else:
        print("[train] dataset adversary: disabled")
    if model.mae_decoder is not None:
        n_mae_params = sum(p.numel() for p in model.mae_decoder.parameters())
        print(
            f"[train] masked autoencoder: enabled "
            f"(patch=({cfg.mae_patch_z}, {cfg.mae_patch_size}, {cfg.mae_patch_size}), "
            f"mask_ratio={cfg.mae_mask_ratio}, params={n_mae_params:,}, "
            f"weight={cfg.mae_weight})"
        )
    else:
        print("[train] masked autoencoder: disabled")
    if model.ibot_enabled:
        # warning: block_size is in FEATURE-GRID cells, not voxels, unlike
        # mae_patch_* above. at layer4 the grid is 8x8 over a 256px crop. one
        # cell is therefore 32px, and the masked block is full-depth x
        # (32*block) x (32*block). the print below gives both units, so nobody
        # tunes ibot_block_size as if it were voxels.
        _cell_px = cfg.global_crop_size // 8
        print(
            f"[train] iBOT masking: enabled "
            f"(block={cfg.ibot_block_size} grid cells on {cfg.ibot_layer} "
            f"= full-depth x {_cell_px * cfg.ibot_block_size}px square, "
            f"mask_ratio={cfg.ibot_mask_ratio}, weight={cfg.ibot_weight})"
        )
    else:
        print("[train] iBOT masking: disabled")
    # 'cpu' here when device says 'cuda' means .to(device) silently did not take.
    print(f"[train] student params live on: {next(model.student.parameters()).device}")
    print(f"[train] teacher params live on: {next(model.teacher.parameters()).device}")

    # ---- optimizer + amp ----
    trainable_modules = torch.nn.ModuleList([model.student])
    if model.dataset_classifier is not None:
        trainable_modules.append(model.dataset_classifier)
    if model.mae_decoder is not None:
        trainable_modules.append(model.mae_decoder)
    optimizer = AdamW(
        get_param_groups(trainable_modules, cfg.weight_decay_start),
        lr=cfg.base_lr,
        betas=(0.9, 0.999),
    )
    scaler = GradScaler(device.type, enabled=use_scaler)

    # ---- io ----
    # one timestamped subdirectory per invocation, so runs stay separate in the
    # tensorboard UI. the `_resume` suffix marks a resume.
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
    ibot_last = [None, None, None, 0]  # [loss, top1, top5, step] from the last masked step

    step = start_step
    for step in range(start_step, cfg.total_steps):
        teach_temp = linear_frac_schedule(
            step, cfg.total_steps, cfg.teacher_temp_start, cfg.teacher_temp_end, cfg.temp_warmup_frac
        )
        model.dataset_adversary_grl_scale = linear_frac_schedule(
            step, cfg.total_steps, cfg.dataset_adversary_grl_start, cfg.dataset_adversary_grl_scale, cfg.temp_warmup_frac
        ) # the same warmup fraction as teacher temp for now. separate it if needed
        # the IterableDataset is infinite. a worker restart can still raise
        # StopIteration, and StopIteration must not kill training.
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        batch = batch_to_device(batch, device)

        # first iteration only. the print catches "model is cuda but batches
        # stay on cpu". that fault silently costs an implicit transfer per op.
        if step == start_step:
            print(f"[train] first batch globals device: {batch['globals'].device}")
            if "locals" in batch:
                print(f"[train] first batch locals  device: {batch['locals'].device}")
            if "weak_label" in batch:
                dataset_indices = batch["weak_label"][:, 0].detach().cpu()
                dataset_counts = torch.bincount(
                    dataset_indices, minlength=n_datasets
                ).tolist()
                print(f"[train] first batch dataset counts: {dataset_counts}")

        lr = lr_schedule(step, cfg.total_steps, cfg.base_lr, cfg.warmup_frac)

        # weight decay reuses the ema cosine schedule with its own start and end.
        wd = ema_momentum_schedule(step, cfg.total_steps, cfg.weight_decay_start, cfg.weight_decay_end)
        for g in optimizer.param_groups:
            g["lr"] = lr
            if g.get("wd_scheduled"): # skip the no-decay group
                g["weight_decay"] = wd

        with autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            out = model(batch)
            dino = dino_loss(
                out["student_globals"],
                out["student_locals"],
                out["teacher_globals"],
                model.center,
                out["B"], out["n_g"], out["n_l"],
                teacher_temp=teach_temp,
                student_temp=cfg.student_temp,
            )
            dataset_loss = None
            if out["dataset_logits"] is not None:
                dataset_loss = F.cross_entropy(
                    out["dataset_logits"], out["dataset_targets"]
                )
            mae_loss = out["mae"]["loss"] if out["mae"] is not None else None
            ibot_loss_val = None
            if out["student_masking_globals"] is not None:
                ibot_loss_val = ibot_patch_loss(
                    out["student_masking_globals"],
                    out["teacher_masking_globals"],
                    out["ibot_mask"],
                    model.ibot_center,
                    teacher_temp=teach_temp,
                    student_temp=cfg.student_temp,
                )
                # the code captures on every masked step, not on log-steps only,
                # so the point lands at its real step. the capture is cheap:
                # argmax and topk, with no backward.
                with torch.no_grad():
                    _ibot_acc = ibot_patch_accuracy(
                        out["student_masking_globals"],
                        out["teacher_masking_globals"],
                        out["ibot_mask"],
                        model.ibot_center,
                    )
                ibot_last = [
                    ibot_loss_val.item(),
                    _ibot_acc["top1"].item(),
                    _ibot_acc["topk"].item(),
                    step,
                ]

            koleo_val = out["koleo"] if out["koleo"] is not None else None
            koleo_weight = None
            if koleo_val is not None:
                koleo_weight = linear_frac_schedule(
                    step, cfg.total_steps, 0.0, cfg.koleo_weight, cfg.temp_warmup_frac
                ) # the same warmup fraction as teacher temp for now. separate it if needed

            loss = dino
            if dataset_loss is not None:
                loss = loss + cfg.dataset_adversary_weight * dataset_loss
            if mae_loss is not None:
                loss = loss + cfg.mae_weight * mae_loss
            if ibot_loss_val is not None:
                loss = loss + cfg.ibot_weight * ibot_loss_val
            if koleo_val is not None:
                loss = loss + koleo_weight * koleo_val

        # unscale before the clip, so the clip sees true-scale gradients. under
        # bf16 the scaler is off, and all four calls are passthroughs.
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        grad_norm = torch.nn.utils.clip_grad_norm_(
            trainable_modules.parameters(), cfg.grad_clip_norm
        )
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

        # EMA and center updates, after the optimizer step.
        m = ema_momentum_schedule(
            step, cfg.total_steps, cfg.ema_momentum_start, cfg.ema_momentum_end
        )
        model.update_teacher(m)
        # raw teacher logits, in float32 for stability.
        model.update_center(out["teacher_globals"].float(), cfg.center_momentum)
        if out["teacher_masking_globals"] is not None:
            model.update_ibot_center(out["teacher_masking_globals"].float(), cfg.center_momentum)

        # ---- logging ----
        if step % cfg.log_interval == 0:
            with torch.no_grad():
                # teacher entropy. read the entropy as a fraction of
                # ln(projector_k). ~0.71 is healthy. a climb toward 1.0 is
                # uniform collapse. 0 is a collapsed teacher. see README
                # "Training".
                t_probs = (
                    (out["teacher_globals"] - model.center) / teach_temp
                ).softmax(dim=-1)
                entropy = -(t_probs * t_probs.clamp(min=1e-8).log()).sum(dim=-1).mean()
                # embedding variance. a value near 0 means all crops map to one
                # point. the code casts to fp32, because under autocast
                # last_features is fp16/bf16. the var of small
                # reduced-precision values underflows exactly when we must see
                # collapse.
                feats = model.student.last_features
                emb_var = (
                    feats.float().var(dim=0).mean()
                    if feats is not None
                    else torch.tensor(0.0)
                )
                center_norm = model.center.norm()
                if dataset_loss is not None:
                    dataset_acc = (
                        out["dataset_logits"].argmax(dim=1)
                        == out["dataset_targets"]
                    ).float().mean()
                else:
                    dataset_acc = None
                mae_mask_ratio = (
                    out["mae"]["mask_ratio"] if out["mae"] is not None else None
                )
                # iBOT metrics come from ibot_last, which the loss block fills.

            dt = time.time() - t_last_log
            t_last_log = time.time()
            steps_per_sec = cfg.log_interval / max(dt, 1e-6) if step > start_step else 0.0

            writer.add_scalar("loss", loss.item(), step)
            writer.add_scalar("loss/dino", dino.item(), step)
            if koleo_val is not None:
                writer.add_scalar("loss/koleo", koleo_val.item(), step)
            writer.add_scalar("lr", lr, step)
            writer.add_scalar("ema_momentum", m, step)
            writer.add_scalar("warmups/teacher_temp", teach_temp, step)
            writer.add_scalar("warmups/dataset_adversary_grl_scale", model.dataset_adversary_grl_scale, step)
            writer.add_scalar("warmups/weight_decay", wd, step)
            if koleo_weight is not None:
                writer.add_scalar("warmups/koleo_weight", koleo_weight, step)
            writer.add_scalar("grad_norm", grad_norm.item(), step)
            writer.add_scalar("collapse/teacher_entropy", entropy.item(), step)
            writer.add_scalar("collapse/embedding_var", emb_var.item(), step)
            writer.add_scalar("collapse/center_norm", center_norm.item(), step)
            writer.add_scalar("throughput/steps_per_sec", steps_per_sec, step)
            if dataset_loss is not None and dataset_acc is not None:
                writer.add_scalar("dataset_adversary/loss", dataset_loss.item(), step)
                writer.add_scalar("dataset_adversary/acc", dataset_acc.item(), step)
            if mae_loss is not None and mae_mask_ratio is not None:
                writer.add_scalar("mae/loss", mae_loss.item(), step)
                writer.add_scalar("mae/mask_ratio", mae_mask_ratio.item(), step)
                writer.add_scalar("mae/recon_r2", out["mae"]["recon_r2"].item(), step)
                writer.add_scalar("mae/recon_corr", out["mae"]["recon_corr"].item(), step)
                writer.add_scalar("mae/recon_rel_l2", out["mae"]["recon_rel_l2"].item(), step)
            if cfg.ibot_enabled:
                if ibot_last[0] is not None:
                    writer.add_scalar("ibot/loss", ibot_last[0], ibot_last[3])
                if ibot_last[1] is not None:
                    writer.add_scalar("ibot/recon_acc_top1", ibot_last[1], ibot_last[3])
                if ibot_last[2] is not None:
                    writer.add_scalar("ibot/recon_acc_top5", ibot_last[2], ibot_last[3])

            msg = (
                f"step {step:06d}  loss={loss.item():.4f}  lr={lr:.2e}  "
                f"teach_temp={teach_temp:.4f}  "
                f"ent={entropy.item():.3f}  emb_var={emb_var.item():.4f}  "
                f"center_norm={center_norm.item():.3f}  {steps_per_sec:.2f} step/s"
            )
            if dataset_loss is not None and dataset_acc is not None:
                msg += (
                    f"  ds_loss={dataset_loss.item():.4f} "
                    f"ds_acc={dataset_acc.item():.3f}"
                )
            if mae_loss is not None:
                msg += (
                    f"  mae_loss={mae_loss.item():.4f} "
                    f"mae_r2={out['mae']['recon_r2'].item():.3f}"
                )
            if ibot_last[0] is not None:
                msg += (
                    f"  ibot_loss={ibot_last[0]:.4f} "
                    f"ibot_acc@1={ibot_last[1]:.3f} ibot_acc@5={ibot_last[2]:.3f}"
                )
            print(msg)

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
