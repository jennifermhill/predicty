"""Run Grad-CAM on behavior and dataset probes.

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

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset, worker_init_fn
from models.encoder import HybridEncoder
from eval.gradcam import GradCAM, get_final_conv_layer, upsample_gradcam
from eval.visualize_gradcam import (
    visualize_gradcam_single,
    visualize_gradcam_multiclass,
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
    return p.parse_args()


def load_behavior_probe(
    checkpoint_path: str,
    latent_dim: int,
    device: torch.device,
) -> tuple[nn.Linear, list[str]]:
    """Load behavior classifier from probe checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    class_names = ckpt["class_names"]

    head = nn.Linear(input_dim, output_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    return head, class_names


def load_dataset_probe(
    checkpoint_path: str,
    latent_dim: int,
    device: torch.device,
) -> tuple[nn.Linear, list[str]]:
    """Load dataset discriminator from probe checkpoint."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    input_dim = ckpt["input_dim"]
    output_dim = ckpt["output_dim"]
    dataset_names = ckpt["dataset_names"]

    head = nn.Linear(input_dim, output_dim).to(device)
    head.load_state_dict(ckpt["head_state_dict"])

    return head, dataset_names


def create_dummy_probes(
    latent_dim: int,
    device: torch.device,
) -> tuple[nn.Linear, nn.Linear, list[str], list[str]]:
    """
    Create dummy probe heads for demonstration.

    In production, load actual trained probes from checkpoints.
    """
    n_behaviors = 4  # streaming, slugs, aggregation, empty
    n_datasets = 20  # Number of zarr datasets

    behavior_head = nn.Linear(latent_dim, n_behaviors).to(device)
    dataset_head = nn.Linear(latent_dim, n_datasets).to(device)

    behavior_names = ["streaming", "slugs", "aggregation", "empty"]
    dataset_names = [f"dataset_{i}" for i in range(n_datasets)]

    return behavior_head, dataset_head, behavior_names, dataset_names


def extract_sample_crops(
    cfg: Config,
    ds: MultiZarrDataset,
    n_samples: int = 10,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Extract sample crops from the dataset.

    Returns:
        List of dicts with keys: 'crop', 'metadata'
    """
    cfg.seed = seed

    # MultiZarrDataset has no __len__/indexing -- it's meant to be iterated
    # via DataLoader, which also works fine for ZarrCellDataset.
    loader = DataLoader(
        ds,
        batch_size=1,
        num_workers=0,
        worker_init_fn=worker_init_fn,
    )

    samples = []
    for batch in loader:
        samples.append(
            {
                "crop": batch["globals"][0, 0],  # (1, Z, H, W) - use t1 global
                "metadata": {k: v[0] for k, v in batch["meta"].items()},
            }
        )
        if len(samples) >= n_samples:
            break
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
) -> None:
    """Run Grad-CAM for behavior classification."""
    print("\n" + "=" * 70)
    print("BEHAVIOR GRAD-CAM")
    print("=" * 70)

    output_dir = output_dir / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Grad-CAM
    target_layer = get_final_conv_layer(encoder)
    gradcam = GradCAM(encoder, behavior_head, target_layer, device)

    samples_df = pd.read_json(cfg.annotation_path, lines=True)
    samples_df = samples_df[samples_df['label'].isin(behavior_names)]
    samples_df = samples_df[samples_df['zarr_path'].isin(cfg.zarr_paths)]
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

        # Generate Grad-CAM for all behaviors
        print(f"\nSample {sample_idx + 1}/{n_samples}")
        gradcams = gradcam.generate_gradcam_multiclass(
            crop,
            len(behavior_names),
        )

        # Upsample to input resolution
        h_in, w_in = crop.shape[-2:]
        gradcams_upsampled = {
            class_idx: upsample_gradcam(hm, (h_in, w_in))
            for class_idx, hm in gradcams.items()
        }

        # Convert crop to numpy for visualization
        crop_np = crop[0].cpu().numpy()  # (1, Z, H, W)
        crop_2d = max_project_3d(crop_np)  # (H, W)

        # Visualize
        gradcams_np = {k: v.cpu().numpy() for k, v in gradcams_upsampled.items()}
        visualize_gradcam_multiclass(
            crop_2d,
            gradcams_np,
            behavior_names,
            title=f"Behavior Grad-CAM - Sample {sample_idx + 1} - Behavior {behavior_name}",
            save_path=str(output_dir / f"behavior_gradcam_{sample_idx:03d}.png"),
        )

        print(f"  Saved: behavior_gradcam_{sample_idx:03d}.png")


def run_dataset_gradcam(
    encoder: HybridEncoder,
    dataset_head: nn.Linear,
    dataset_names: list[str],
    samples: list[dict],
    device: torch.device,
    output_dir: Path,
) -> None:
    """Run Grad-CAM for dataset discrimination."""
    print("\n" + "=" * 70)
    print("DATASET DISCRIMINATION GRAD-CAM")
    print("=" * 70)

    output_dir = output_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup Grad-CAM
    target_layer = get_final_conv_layer(encoder)
    gradcam = GradCAM(encoder, dataset_head, target_layer, device)
    for sample_idx, sample in enumerate(samples):
        crop = sample["crop"]  # (C=1, Z, H, W), no batch dim
        crop = crop.unsqueeze(0).to(device)  # (1, 1, Z, H, W) = (B, C, Z, H, W)

        # Generate Grad-CAM for all datasets (expensive - sample a few)
        print(f"\nSample {sample_idx + 1}/{len(samples)}")
        n_datasets_to_show = min(5, len(dataset_names))
        gradcams = {}

        for dataset_idx in range(n_datasets_to_show):
            hm = gradcam.generate_gradcam(crop, dataset_idx)
            gradcams[dataset_idx] = hm

        # Upsample
        h_in, w_in = crop.shape[-2:]
        gradcams_upsampled = {
            class_idx: upsample_gradcam(hm, (h_in, w_in))
            for class_idx, hm in gradcams.items()
        }

        # Visualize
        crop_np = crop[0].cpu().numpy()
        crop_2d = max_project_3d(crop_np)

        gradcams_np = {k: v.cpu().numpy() for k, v in gradcams_upsampled.items()}
        dataset_names_subset = dataset_names[:n_datasets_to_show]
        for i in range(len(dataset_names_subset)):
            dataset_names_subset[i] = dataset_names_subset[i][dataset_names_subset[i].rindex("/") + 1:]
            # strips path for better visualization

        visualize_gradcam_multiclass(
            crop_2d,
            gradcams_np,
            dataset_names_subset,
            title=f"Dataset Discrimination Grad-CAM - Sample {sample_idx + 1}",
            save_path=str(output_dir / f"dataset_gradcam_{sample_idx:03d}.png"),
        )

        print(f"  Saved: dataset_gradcam_{sample_idx:03d}.png")


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
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()

    print(f"[gradcam] encoder loaded from step {ckpt['step']}")

    # Auto-detect probe checkpoints from encoder directory
    encoder_dir = Path(args.checkpoint).parent
    if args.behavior_probe is None:
        args.behavior_probe = encoder_dir / "behavior_probe.pt"
    if args.dataset_probe is None:
        args.dataset_probe = encoder_dir / "dataset_probe.pt"

    # Load probes
    behavior_head = None
    dataset_head = None
    behavior_names = None
    dataset_names = None

    if args.mode in ["behavior", "both"]:
        if not Path(args.behavior_probe).exists():
            print(f"[gradcam] WARNING: behavior probe not found at {args.behavior_probe}")
            print(f"[gradcam] Run: python eval/probe_behavior.py --config {args.config} --checkpoint {args.checkpoint}")
        else:
            print(f"[gradcam] loading behavior probe from {args.behavior_probe}")
            behavior_head, behavior_names = load_behavior_probe(
                str(args.behavior_probe), cfg.latent_dim, device
            )
            print(f"[gradcam] behavior classes: {behavior_names}")

    if args.mode in ["dataset", "both"]:
        if not Path(args.dataset_probe).exists():
            print(f"[gradcam] WARNING: dataset probe not found at {args.dataset_probe}")
            print(f"[gradcam] Run: python eval/probe_dataset.py --config {args.config} --checkpoint {args.checkpoint}")
        else:
            print(f"[gradcam] loading dataset probe from {args.dataset_probe}")
            dataset_head, dataset_names = load_dataset_probe(
                str(args.dataset_probe), cfg.latent_dim, device
            )
            print(f"[gradcam] dataset classes: {len(dataset_names)} datasets")

    # Extract sample crops
    ds = MultiZarrDataset(cfg)
    print(f"[gradcam] extracting {args.n_samples} sample crops...")
    samples = extract_sample_crops(cfg, ds, args.n_samples, args.seed)
    print(f"[gradcam] extracted {len(samples)} samples")

    # Output directory
    output_dir = Path("latent_space") / f"gradcam_{time.strftime('%Y%m%d_%H%M%S')}"

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
        )
    elif args.mode in ["dataset", "both"]:
        print("\n[gradcam] Skipping dataset Grad-CAM (probe not loaded)")

    print("\n" + "=" * 70)
    print(f"[gradcam] visualizations saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
