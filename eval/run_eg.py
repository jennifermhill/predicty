"""the script runs expected gradients on the behavior probe and the dataset probe.

the script is the input-space companion to run_gradcam.py. the maps are at input
resolution, and they keep the z axis that grad-cam discards. see
eval/expected_gradients.py for the reason to use EG rather than fixed-baseline
integrated gradients on this encoder.

usage (--mode behavior | dataset | both; --z-slices adds per-z panels):
    python eval/run_eg.py --config configs/train.yaml --checkpoint checkpoints/latest.pt --mode both
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import torch

from configs.config import Config
from data.dataset import MultiZarrDataset
from models.encoder import HybridEncoder
from eval.expected_gradients import (
    ExpectedGradients,
    reduce_over_z,
    visualize_eg_multiclass,
    visualize_eg_zslices,
)
from eval.visualize_gradcam import max_project_3d

# the grad-cam runner shares these functions, so that both scripts explain crops
# that they load in the same way.
from eval.run_gradcam import (
    load_behavior_probe,
    load_dataset_probe,
    extract_sample_crops,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to training config yaml")
    p.add_argument("--checkpoint", required=True, help="path to encoder checkpoint")
    p.add_argument(
        "--behavior-probe",
        default=None,
        help="path to behavior probe checkpoint (auto-detected from latent_space if unset)",
    )
    p.add_argument(
        "--dataset-probe",
        default=None,
        help="path to dataset probe checkpoint (auto-detected from latent_space if unset)",
    )
    p.add_argument(
        "--mode",
        choices=["behavior", "dataset", "both"],
        default="both",
        help="which probes to visualize",
    )
    p.add_argument("--device", default=None, help="override device (e.g., cpu)")
    p.add_argument(
        "--n-crops", type=int, default=10, help="crops to visualize per mode"
    )
    p.add_argument(
        "--n-samples",
        type=int,
        default=128,
        help="expected-gradients monte-carlo draws per attribution",
    )
    p.add_argument(
        "--baseline-pool-size",
        type=int,
        default=32,
        help="real crops sampled to form the baseline distribution",
    )
    p.add_argument(
        "--chunk-size",
        type=int,
        default=8,
        help="draws per forward/backward; lower if globals OOM",
    )
    p.add_argument(
        "--z-reduce",
        choices=["sum", "max_abs"],
        default="sum",
        help="how to collapse per-voxel attribution to a 2D heatmap",
    )
    p.add_argument(
        "--z-slices",
        action="store_true",
        help="also save per-z-slice panels for the predicted class",
    )
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _predict(encoder, head, crop, n_classes):
    """the predicted class and the probability dict for a crop."""
    with torch.no_grad():
        prob_vec = torch.softmax(head(encoder(crop)), dim=-1)[0]
        pred_class = int(prob_vec.argmax().item())
        probs = {i: float(prob_vec[i].item()) for i in range(n_classes)}
    return pred_class, probs


def build_baseline_pool(
    ds: MultiZarrDataset, pool_size: int, device: torch.device
) -> torch.Tensor:
    """the function stacks real crops into a baseline distribution, shape (M, 1, Z, H, W)."""
    pool_samples = extract_sample_crops(ds, pool_size)
    if not pool_samples:
        raise ValueError("could not sample any crops for the baseline pool")
    baselines = torch.stack([s["crop"] for s in pool_samples]).to(device)
    return baselines


def run_behavior_eg(
    cfg: Config,
    ds: MultiZarrDataset,
    encoder: HybridEncoder,
    behavior_head,
    behavior_names,
    eg: ExpectedGradients,
    baselines: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 70)
    print("BEHAVIOR EXPECTED GRADIENTS")
    print("=" * 70)

    output_dir = output_dir / "behavior"
    output_dir.mkdir(parents=True, exist_ok=True)

    # the per-behavior sample selection is the same as in run_gradcam, so that
    # the two methods explain comparable inputs.
    samples_df = pd.read_json(cfg.annotation_path, lines=True)
    samples_df = samples_df[samples_df["label"].isin(behavior_names)]
    if cfg.zarr_paths is None:
        raise ValueError("cfg.zarr_paths must be set for behavior EG with annotations")
    samples_df = samples_df[samples_df["zarr_path"].isin(cfg.zarr_paths)]
    if len(samples_df) == 0:
        raise ValueError("No annotated samples found matching behavior names and zarr paths")
    samples_df = samples_df.drop(columns=["timestamp"])
    for col in samples_df.columns:
        try:
            samples_df[col] = pd.to_numeric(samples_df[col])
        except (ValueError, TypeError):
            pass

    for crop_idx in range(args.n_crops):
        behavior_idx = crop_idx % len(behavior_names)
        behavior_name = behavior_names[behavior_idx]
        sample = (
            samples_df[samples_df["label"] == behavior_name]
            .sample(n=1, random_state=cfg.seed + crop_idx)
            .iloc[0]
        )
        zarr_index = cfg.zarr_paths.index(sample["zarr_path"])
        src = ds.children[zarr_index]
        batch = src._annotation_tensor(
            t1=sample["t1"],
            t2=sample["t2"],
            z=sample["z"],
            global_x=sample["global_x"],
            global_y=sample["global_y"],
        )
        crop = batch["globals"][0].unsqueeze(0).to(device)  # (1, 1, Z, H, W)

        pred_class, probs = _predict(encoder, behavior_head, crop, len(behavior_names))

        print(f"\nCrop {crop_idx + 1}/{args.n_crops} (true: {behavior_name})")
        results = eg.attribute_multiclass(
            crop,
            len(behavior_names),
            baselines,
            n_samples=args.n_samples,
            chunk_size=args.chunk_size,
            seed=args.seed + crop_idx,
        )

        heatmaps = {
            k: reduce_over_z(r["attribution"], args.z_reduce)
            for k, r in results.items()
        }
        completeness = {k: r["completeness"] for k, r in results.items()}
        print(
            f"  completeness error (pred class): "
            f"{completeness[pred_class] * 100:.2f}%"
        )

        crop_np = crop[0, 0].cpu().numpy()  # (Z, H, W)
        crop_2d = max_project_3d(crop_np)

        visualize_eg_multiclass(
            crop_2d,
            heatmaps,
            behavior_names,
            title=(
                f"Behavior EG - Crop {crop_idx + 1} - "
                f"true: {behavior_name} - pred: {behavior_names[pred_class]}"
            ),
            save_path=str(output_dir / f"behavior_eg_{crop_idx:03d}.png"),
            pred_class=pred_class,
            probs=probs,
            completeness=completeness,
        )
        if args.z_slices:
            visualize_eg_zslices(
                crop_np,
                results[pred_class]["attribution"],
                behavior_names[pred_class],
                save_path=str(
                    output_dir / f"behavior_eg_{crop_idx:03d}_zslices.png"
                ),
            )
        print(f"  Saved: behavior_eg_{crop_idx:03d}.png")


def run_dataset_eg(
    encoder: HybridEncoder,
    dataset_head,
    dataset_names,
    eg: ExpectedGradients,
    baselines: torch.Tensor,
    samples,
    args: argparse.Namespace,
    device: torch.device,
    output_dir: Path,
) -> None:
    print("\n" + "=" * 70)
    print("DATASET DISCRIMINATION EXPECTED GRADIENTS")
    print("=" * 70)

    output_dir = output_dir / "dataset"
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_names_clean = [
        name.rsplit("/", 1)[-1].replace(".zarr", "") for name in dataset_names
    ]
    n_show = min(5, len(dataset_names))

    for crop_idx, sample in enumerate(samples):
        crop = sample["crop"].unsqueeze(0).to(device)  # (1, 1, Z, H, W)
        own_idx = int(sample["metadata"]["dataset_idx"])

        # the list always starts with the crop's own dataset. the other datasets
        # then fill the list.
        shown_idx = [own_idx]
        for idx in range(len(dataset_names)):
            if len(shown_idx) >= n_show:
                break
            if idx != own_idx:
                shown_idx.append(idx)

        pred_idx, _ = _predict(encoder, dataset_head, crop, len(dataset_names))

        print(
            f"\nCrop {crop_idx + 1}/{len(samples)} "
            f"(from {dataset_names_clean[own_idx]}, predicted {dataset_names_clean[pred_idx]})"
        )
        results = {
            idx: eg.attribute(
                crop,
                idx,
                baselines,
                n_samples=args.n_samples,
                chunk_size=args.chunk_size,
                seed=args.seed + crop_idx,
            )
            for idx in shown_idx
        }
        heatmaps = {
            k: reduce_over_z(r["attribution"], args.z_reduce)
            for k, r in results.items()
        }
        completeness = {k: r["completeness"] for k, r in results.items()}
        print(
            f"  completeness error (pred class): "
            f"{completeness.get(pred_idx, float('nan')) * 100:.2f}%"
        )

        crop_np = crop[0, 0].cpu().numpy()
        crop_2d = max_project_3d(crop_np)

        visualize_eg_multiclass(
            crop_2d,
            heatmaps,
            dataset_names_clean,
            title=(
                f"Dataset EG - Crop {crop_idx + 1} - "
                f"crop from {dataset_names_clean[own_idx]}"
            ),
            save_path=str(output_dir / f"dataset_eg_{crop_idx:03d}.png"),
            pred_class=pred_idx,
            completeness=completeness,
        )
        if args.z_slices:
            visualize_eg_zslices(
                crop_np,
                results[pred_idx]["attribution"],
                dataset_names_clean[pred_idx],
                save_path=str(
                    output_dir / f"dataset_eg_{crop_idx:03d}_zslices.png"
                ),
            )
        print(f"  Saved: dataset_eg_{crop_idx:03d}.png")


def main() -> None:
    args = parse_args()

    cfg = Config.from_yaml(args.config)
    device = torch.device(args.device or cfg.device)

    print(f"[eg] device: {device}")
    print(f"[eg] config: {args.config}")
    print(f"[eg] checkpoint: {args.checkpoint}")
    print(
        f"[eg] mode: {args.mode}  n_samples: {args.n_samples}  "
        f"baseline_pool: {args.baseline_pool_size}  chunk: {args.chunk_size}"
    )
    print()

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "encoder_state_dict" not in ckpt:
        print("[eg] ERROR: checkpoint missing 'encoder_state_dict'. invalid format.")
        return

    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    encoder.eval()
    print(f"[eg] encoder loaded from step {ckpt.get('step', 'unknown')}")

    # warning: the sort must use mtime and never the name. the probe
    # auto-detection is the same as in run_gradcam.main. see the comment there
    # for the reason.
    run_name = Path(args.checkpoint).parent.name
    latent_dir = Path("latent_space")

    def _autodetect_probe(kind: str):
        if not latent_dir.exists():
            return None
        hits = sorted(latent_dir.glob(f"{run_name}*/{kind}.pt"),
                      key=lambda p: p.stat().st_mtime, reverse=True)
        if not hits:  # no probe for this run yet, so take the newest one anywhere
            hits = sorted(latent_dir.glob(f"*/{kind}.pt"),
                          key=lambda p: p.stat().st_mtime, reverse=True)
        return hits[0] if hits else None

    if args.behavior_probe is None:
        args.behavior_probe = _autodetect_probe("behavior_probe")
    if args.dataset_probe is None:
        args.dataset_probe = _autodetect_probe("dataset_probe")

    behavior_head = behavior_names = None
    dataset_head = dataset_names = None

    if args.mode in ["behavior", "both"]:
        if args.behavior_probe is None or not Path(args.behavior_probe).exists():
            print(f"[eg] WARNING: behavior probe not found at {args.behavior_probe}")
        else:
            print(f"[eg] loading behavior probe from {args.behavior_probe}")
            behavior_head, behavior_names = load_behavior_probe(
                str(args.behavior_probe), device
            )
            print(f"[eg] behavior classes: {behavior_names}")

    if args.mode in ["dataset", "both"]:
        if args.dataset_probe is None or not Path(args.dataset_probe).exists():
            print(f"[eg] WARNING: dataset probe not found at {args.dataset_probe}")
        else:
            print(f"[eg] loading dataset probe from {args.dataset_probe}")
            dataset_head, dataset_names = load_dataset_probe(
                str(args.dataset_probe), device
            )
            print(f"[eg] dataset classes: {len(dataset_names)} datasets")

    cfg.seed = args.seed
    ds = MultiZarrDataset(cfg)

    print(f"[eg] building baseline pool ({args.baseline_pool_size} crops)...")
    baselines = build_baseline_pool(ds, args.baseline_pool_size, device)
    print(f"[eg] baseline pool: {tuple(baselines.shape)}")

    print(f"[eg] extracting {args.n_crops} crops to visualize...")
    samples = extract_sample_crops(ds, args.n_crops)
    print(f"[eg] extracted {len(samples)} crops")
    if not samples:
        print("[eg] WARNING: no crops extracted. check dataset configuration.")
        return

    # the outputs go into this run's own latent_space folder, next to the probes.
    run_dirs = sorted(
        (p for p in latent_dir.glob(f"{run_name}*") if p.is_dir()),
        key=lambda p: p.stat().st_mtime, reverse=True,
    )
    base = run_dirs[0] if run_dirs else latent_dir / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = base / "eg"
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode in ["behavior", "both"] and behavior_head is not None:
        eg = ExpectedGradients(encoder, behavior_head, device)
        run_behavior_eg(
            cfg, ds, encoder, behavior_head, behavior_names, eg,
            baselines, args, device, output_dir,
        )
    elif args.mode in ["behavior", "both"]:
        print("\n[eg] Skipping behavior EG (probe not loaded)")

    if args.mode in ["dataset", "both"] and dataset_head is not None:
        eg = ExpectedGradients(encoder, dataset_head, device)
        run_dataset_eg(
            encoder, dataset_head, dataset_names, eg,
            baselines, samples, args, device, output_dir,
        )
    elif args.mode in ["dataset", "both"]:
        print("\n[eg] Skipping dataset EG (probe not loaded)")

    print("\n" + "=" * 70)
    print(f"[eg] visualizations saved to {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
