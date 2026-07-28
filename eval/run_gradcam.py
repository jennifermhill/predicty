"""run Grad-CAM on behavior and dataset probes.

usage:
    # Behavior Grad-CAM
    python eval/run_gradcam.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --mode behavior

    # Dataset discrimination Grad-CAM
    python eval/run_gradcam.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --mode dataset

    # Both
    python eval/run_gradcam.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --mode both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch
import torch.nn as nn

from configs.config import Config
from data.dataset import MultiZarrDataset
from models.encoder import HybridEncoder
from eval.gradcam import (
    GradCAMAnalyzer,
    upsample_gradcam,
)
from eval.visualize_gradcam import (
    visualize_gradcam_multiclass,
    visualize_layercam_3d,
    max_project_3d,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to training config yaml")
    p.add_argument("--checkpoint", required=True, help="path to encoder checkpoint")
    p.add_argument(
        "--behavior-probe",
        default=None,
        help="path to behavior probe checkpoint (auto-detected from encoder dir if not specified)",
    )
    p.add_argument(
        "--dataset-probe",
        default=None,
        help="path to dataset probe checkpoint (auto-detected from encoder dir if not specified)",
    )
    p.add_argument(
        "--mode",
        choices=["behavior", "dataset", "both"],
        default="both",
        help="which probes to visualize",
    )
    p.add_argument("--device", default=None, help="override device (e.g., cpu)")
    p.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="number of samples to visualize per class",
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--output-dir",
        default=None,
        help="where to save visualizations (default: latent_space/gradcam_<timestamp>)",
    )
    p.add_argument(
        "--cam-3d",
        action="store_true",
        help=(
            "also produce a full-resolution 3D LayerCAM per sample at the 3D "
            "stem (before z is collapsed), saved as a per-z-slice montage"
        ),
    )
    return p.parse_args()


def load_behavior_probe(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Linear, list[str]]:
    """load behavior classifier from probe checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    required_keys = ["input_dim", "output_dim", "class_names", "head_state_dict"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise ValueError(f"Probe checkpoint missing keys: {missing}")

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    class_names = ckpt["class_names"]

    head = nn.Linear(input_dim, output_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    return head, class_names


def load_dataset_probe(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[nn.Linear, list[str]]:
    """load dataset discriminator from probe checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    required_keys = ["input_dim", "output_dim", "dataset_names", "head_state_dict"]
    missing = [k for k in required_keys if k not in ckpt]
    if missing:
        raise ValueError(f"Probe checkpoint missing keys: {missing}")

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    dataset_names = ckpt["dataset_names"]

    head = nn.Linear(input_dim, output_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])
    head.eval()

    return head, dataset_names


def extract_sample_crops(
    ds: MultiZarrDataset,
    n_samples: int = 10,
) -> list[dict[str, Any]]:
    """
    extract sample crops from an already-constructed dataset.

    Returns:
        list of dicts with keys: 'crop', 'metadata'
    """
    samples = []
    for sample in ds:
        if len(samples) >= n_samples:
            break
        samples.append(
            {
                "crop": sample["globals"][0],  # (1, Z, H, W) - use t1 global
                "metadata": sample["meta"],
            }
        )
    return samples


def run_behavior_gradcam(
    cfg: Config,
    ds: MultiZarrDataset,
    encoder: HybridEncoder,
    behavior_head: nn.Linear,
    behavior_names: list[str],
    n_samples: int,
    device: torch.device,
    output_dir: Path,
    cam_3d: bool = False,
) -> None:
    """run Grad-CAM for behavior classification."""
    print("\n" + "=" * 70)
    print("BEHAVIOR GRAD-CAM")
    print("=" * 70)

    output_dir = output_dir / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    # single-forward analyzer: analytic 2D grad-cam (no backprop) + probs +
    # depth weights from one pass, plus an optional 3D LayerCAM (one backward).
    analyzer = GradCAMAnalyzer(encoder, behavior_head, device, cam_3d=cam_3d)

    samples_df = pd.read_json(cfg.annotation_path, lines=True)
    samples_df = samples_df[samples_df['label'].isin(behavior_names)]
    if cfg.zarr_paths is None:
        raise ValueError("cfg.zarr_paths must be set for behavior Grad-CAM with annotations")
    samples_df = samples_df[samples_df['zarr_path'].isin(cfg.zarr_paths)]
    if len(samples_df) == 0:
        raise ValueError("No annotated samples found matching behavior names and zarr paths")
    samples_df = samples_df.drop(columns = ['timestamp'])
    for col in samples_df.columns:
        try:
            samples_df[col] = pd.to_numeric(samples_df[col])
        except (ValueError, TypeError):
            pass
    for sample_idx in range(n_samples):
        behavior_idx = sample_idx % len(behavior_names)
        behavior_name = behavior_names[behavior_idx]
        sample = samples_df[samples_df['label'] == behavior_name].sample(n=1, random_state=cfg.seed + sample_idx).iloc[0]
        zarr_index = cfg.zarr_paths.index(sample['zarr_path'])
        src = ds.children[zarr_index]
        batch = src._annotation_tensor(
            t1=sample["t1"], 
            t2=sample["t2"], 
            z=sample["z"], 
            global_x=sample["global_x"], 
            global_y=sample["global_y"],
        )

        crop = batch["globals"][0]  # (C=1, Z, H, W), no batch dim
        crop = crop.unsqueeze(0).to(device)  # (1, 1, Z, H, W) = (B, C, Z, H, W)

        # one forward yields 2D grad-cams for every behavior, the prediction +
        # probabilities, depth weights, and (when cam_3d) the 3D map for the
        # true behavior. plot flags what the model decided alongside where it
        # looked.
        print(f"\nSample {sample_idx + 1}/{n_samples}")
        result = analyzer.analyze(
            crop,
            len(behavior_names),
            cam3d_class=behavior_idx if cam_3d else None,
        )
        pred_class = result["pred_class"]
        probs = result["probs"]
        gradcams = result["cams_2d"]

        # Upsample to input resolution
        h_in, w_in = crop.shape[-2:]
        gradcams_upsampled = {
            class_idx: upsample_gradcam(hm, (h_in, w_in))
            for class_idx, hm in gradcams.items()
        }

        # Convert crop to numpy for visualization
        crop_np = crop[0, 0].cpu().numpy()  # (Z, H, W)
        crop_2d = max_project_3d(crop_np)  # (H, W)

        # Visualize
        gradcams_np = {k: v.cpu().numpy() for k, v in gradcams_upsampled.items()}
        visualize_gradcam_multiclass(
            crop_2d,
            gradcams_np,
            behavior_names,
            title=(
                f"Behavior Grad-CAM - Sample {sample_idx + 1} - "
                f"true: {behavior_name} - pred: {behavior_names[pred_class]}"
            ),
            save_path=str(output_dir / f"behavior_gradcam_{sample_idx:03d}.png"),
            pred_class=pred_class,
            probs=probs,
        )

        print(f"  Saved: behavior_gradcam_{sample_idx:03d}.png")

        # full-resolution 3D LayerCAM for the true behavior, shown per z-slice,
        # with the pool's literal depth weighting as a companion bar strip. both
        # came from the same analyze() call above (no extra forward passes).
        if cam_3d:
            cam3d = result["cam_3d"].numpy()  # (Z, H, W)
            z_w = result["z_weights"].numpy()  # (Z,)
            visualize_layercam_3d(
                crop_np,
                cam3d,
                title=(
                    f"Behavior 3D LayerCAM - Sample {sample_idx + 1} - "
                    f"true: {behavior_name}"
                ),
                save_path=str(output_dir / f"behavior_layercam3d_{sample_idx:03d}.png"),
                z_weights=z_w,
            )
            print(f"  Saved: behavior_layercam3d_{sample_idx:03d}.png")


def run_dataset_gradcam(
    encoder: HybridEncoder,
    dataset_head: nn.Linear,
    dataset_names: list[str],
    samples: list[dict],
    device: torch.device,
    output_dir: Path,
    cam_3d: bool = False,
) -> None:
    """run Grad-CAM for dataset discrimination."""
    print("\n" + "=" * 70)
    print("DATASET DISCRIMINATION GRAD-CAM")
    print("=" * 70)

    output_dir = output_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # single-forward analyzer: analytic 2D grad-cam (no backprop) + probs +
    # depth weights from one pass, plus an optional 3D LayerCAM (one backward).
    analyzer = GradCAMAnalyzer(encoder, dataset_head, device, cam_3d=cam_3d)

    # strip directory + .zarr so panel labels read cleanly. the gradcam dict
    # keys are dataset indices used to index this list, so it must stay full
    # length and index-aligned with dataset_names.
    dataset_names_clean = [
        name.rsplit("/", 1)[-1].replace(".zarr", "") for name in dataset_names
    ]
    n_show = min(5, len(dataset_names))

    for sample_idx, sample in enumerate(samples):
        crop = sample["crop"]  # (C=1, Z, H, W), no batch dim
        crop = crop.unsqueeze(0).to(device)  # (1, 1, Z, H, W) = (B, C, Z, H, W)

        own_idx = int(sample["metadata"]["dataset_idx"])

        # always show the crop's own dataset, then fill the panel with others.
        # guarantees the true source dataset appears and avoids duplicate keys.
        shown_idx = [own_idx]
        for idx in range(len(dataset_names)):
            if len(shown_idx) >= n_show:
                break
            if idx != own_idx:
                shown_idx.append(idx)

        # one forward yields the prediction, a 2D grad-cam for every dataset,
        # depth weights, and (when cam_3d) the 3D map for the crop's own dataset.
        result = analyzer.analyze(
            crop,
            len(dataset_names),
            cam3d_class=own_idx if cam_3d else None,
        )
        pred_idx = result["pred_class"]

        print(
            f"\nSample {sample_idx + 1}/{len(samples)} "
            f"(from {dataset_names_clean[own_idx]}, predicted {dataset_names_clean[pred_idx]})"
        )
        # all classes are computed; keep only the panel's shown datasets.
        gradcams = {idx: result["cams_2d"][idx] for idx in shown_idx}

        # Upsample
        h_in, w_in = crop.shape[-2:]
        gradcams_upsampled = {
            class_idx: upsample_gradcam(hm, (h_in, w_in))
            for class_idx, hm in gradcams.items()
        }

        # Visualize
        crop_np = crop[0, 0].cpu().numpy()  # (Z, H, W)
        crop_2d = max_project_3d(crop_np)  # (H, W)

        gradcams_np = {k: v.cpu().numpy() for k, v in gradcams_upsampled.items()}
        visualize_gradcam_multiclass(
            crop_2d,
            gradcams_np,
            dataset_names_clean,
            title=(
                f"Dataset Grad-CAM - Sample {sample_idx + 1} - "
                f"crop from {dataset_names_clean[own_idx]}"
            ),
            save_path=str(output_dir / f"dataset_gradcam_{sample_idx:03d}.png"),
            pred_class=pred_idx,
        )

        print(f"  Saved: dataset_gradcam_{sample_idx:03d}.png")

        # full-resolution 3D LayerCAM for the crop's own dataset, per z-slice,
        # with the pool's literal depth weighting as a companion bar strip. both
        # came from the same analyze() call above (no extra forward passes).
        if cam_3d:
            cam3d = result["cam_3d"].numpy()  # (Z, H, W)
            z_w = result["z_weights"].numpy()  # (Z,)
            visualize_layercam_3d(
                crop_np,
                cam3d,
                title=(
                    f"Dataset 3D LayerCAM - Sample {sample_idx + 1} - "
                    f"crop from {dataset_names_clean[own_idx]}"
                ),
                save_path=str(output_dir / f"dataset_layercam3d_{sample_idx:03d}.png"),
                z_weights=z_w,
            )
            print(f"  Saved: dataset_layercam3d_{sample_idx:03d}.png")


def main() -> None:
    args = parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device or cfg.device)

    print(f"[gradcam] device: {device}")
    print(f"[gradcam] config: {args.config}")
    print(f"[gradcam] checkpoint: {args.checkpoint}")
    print(f"[gradcam] mode: {args.mode}")
    print()

    # Load encoder
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "encoder_state_dict" not in ckpt:
        print(f"[gradcam] ERROR: checkpoint missing 'encoder_state_dict'. invalid checkpoint format.")
        return

    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    print(f"[gradcam] encoder loaded from step {ckpt.get('step', 'unknown')}")

    # Auto-detect probe checkpoints from the latent_space directory. prefer the
    # probe from THIS run's own dir (its input_dim matches this encoder); fall back
    # to the most recently MODIFIED probe. NB: a plain name sort is wrong here --
    # letter-prefixed run names (e.g. "final_...") sort above digit-prefixed ones,
    # so `sorted(..., reverse=True)` would grab a stale probe of the wrong latent_dim.
    run_name = Path(args.checkpoint).parent.name
    latent_dir = Path("latent_space")

    def _autodetect_probe(kind: str):
        if not latent_dir.exists():
            return None
        matches = sorted(latent_dir.glob(f"{run_name}*/{kind}.pt"),
                         key=lambda p: p.stat().st_mtime, reverse=True)
        if not matches:  # no probe for this run yet; fall back to newest anywhere
            matches = sorted(latent_dir.glob(f"*/{kind}.pt"),
                             key=lambda p: p.stat().st_mtime, reverse=True)
        return matches[0] if matches else None

    if args.behavior_probe is None:
        args.behavior_probe = _autodetect_probe("behavior_probe")
    if args.dataset_probe is None:
        args.dataset_probe = _autodetect_probe("dataset_probe")

    # Load probes
    behavior_head = None
    dataset_head = None
    behavior_names = None
    dataset_names = None

    if args.mode in ["behavior", "both"]:
        if args.behavior_probe is None or not Path(args.behavior_probe).exists():
            print(f"[gradcam] WARNING: behavior probe not found at {args.behavior_probe}")
            print(f"[gradcam] Run: python eval/probe_behavior.py --config {args.config} --checkpoint {args.checkpoint}")
        else:
            print(f"[gradcam] loading behavior probe from {args.behavior_probe}")
            behavior_head, behavior_names = load_behavior_probe(
                str(args.behavior_probe), device
            )
            print(f"[gradcam] behavior classes: {behavior_names}")

    if args.mode in ["dataset", "both"]:
        if args.dataset_probe is None or not Path(args.dataset_probe).exists():
            print(f"[gradcam] WARNING: dataset probe not found at {args.dataset_probe}")
            print(f"[gradcam] Run: python eval/probe_dataset.py --config {args.config} --checkpoint {args.checkpoint}")
        else:
            print(f"[gradcam] loading dataset probe from {args.dataset_probe}")
            dataset_head, dataset_names = load_dataset_probe(
                str(args.dataset_probe), device
            )
            print(f"[gradcam] dataset classes: {len(dataset_names)} datasets")

    # Extract sample crops. set the seed before building the dataset so
    # sampling is reproducible.
    cfg.seed = args.seed
    ds = MultiZarrDataset(cfg)
    print(f"[gradcam] extracting {args.n_samples} sample crops...")
    samples = extract_sample_crops(ds, args.n_samples)
    print(f"[gradcam] extracted {len(samples)} samples")

    if not samples:
        print("[gradcam] WARNING: no samples extracted. check dataset configuration or zarr paths.")
        return

    # Output directory. defaults to a "gradcam" subfolder inside THIS run's own
    # latent_space folder (next to the probes), so outputs stay grouped per run
    # and the redundant-run check can find them. an explicit --output-dir wins.
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        run_dirs = sorted(
            (p for p in latent_dir.glob(f"{run_name}*") if p.is_dir()),
            key=lambda p: p.stat().st_mtime, reverse=True,
        )
        base = run_dirs[0] if run_dirs else latent_dir / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        output_dir = base / "gradcam"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Run Grad-CAM
    if args.mode in ["behavior", "both"] and behavior_head is not None:
        run_behavior_gradcam(
            cfg,
            ds,
            encoder,
            behavior_head,
            behavior_names,
            args.n_samples,
            device,
            output_dir,
            cam_3d=args.cam_3d,
        )
    elif args.mode in ["behavior", "both"]:
        print("\n[gradcam] Skipping behavior Grad-CAM (probe not loaded)")

    if args.mode in ["dataset", "both"] and dataset_head is not None:
        run_dataset_gradcam(
            encoder,
            dataset_head,
            dataset_names,
            samples,
            device,
            output_dir,
            cam_3d=args.cam_3d,
        )
    elif args.mode in ["dataset", "both"]:
        print("\n[gradcam] Skipping dataset Grad-CAM (probe not loaded)")

    print("\n" + "=" * 70)
    print(f"[gradcam] visualizations saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
