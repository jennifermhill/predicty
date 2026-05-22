from __future__ import annotations

import argparse
import time
from statistics import median

from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import MultiZarrDataset, ZarrCellDataset, worker_init_fn
from data.picker import pick_zarr_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="optional path to a yaml config (e.g. configs/train.yaml).",
    )
    # peek only at --config; picker.py reads --zarr-path itself.
    args, _ = parser.parse_known_args()
    return args


def main() -> None:
    args = _parse_args()
    cfg = Config.from_yaml(args.config) if args.config else Config()

    # multi-zarr training takes precedence; single-zarr falls back to the picker.
    if cfg.zarr_paths:
        if cfg.zarr_path is not None:
            print(
                "[main] cfg.zarr_paths is set; ignoring cfg.zarr_path "
                f"({cfg.zarr_path!r})."
            )
        ds: ZarrCellDataset | MultiZarrDataset = MultiZarrDataset(cfg)
    else:
        if cfg.zarr_path is None:
            cfg.zarr_path = pick_zarr_path()
        ds = ZarrCellDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=4,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
    )

    per_crop_ms: list[float] = []
    n_batches = 20
    print(f"[main] pulling {n_batches} batches of 4 pairs ...")

    t_loop = time.time()
    for i, batch in zip(range(n_batches), loader):
        t0 = time.time()
        v1 = batch["view1"]
        v2 = batch["view2"]
        meta = batch["meta"]
        dt = (time.time() - t0) * 1000.0
        # 4 pairs means 8 crops per batch.
        per_crop_ms.append(dt / 8.0)
        # dataset_idx is only present when running on a multi-zarr dataset.
        ds_idx = meta["dataset_idx"].tolist() if "dataset_idx" in meta else None
        print(
            f"[main] batch {i:02d} view1={tuple(v1.shape)} view2={tuple(v2.shape)} "
            f"t1={meta['t1'].tolist()} t2={meta['t2'].tolist()} "
            f"z={meta['z'].tolist()} z_end={meta['z_end'].tolist()}"
            + (f" dataset_idx={ds_idx}" if ds_idx is not None else "")
        )

    total = time.time() - t_loop
    if per_crop_ms:
        print(
            f"[main] median ms/crop (post-fetch handling only): "
            f"{median(per_crop_ms):.2f}"
        )
    n_crops = n_batches * 4 * 2
    print(f"[main] total wall time: {total:.2f}s for {n_crops} crops "
          f"-> {total * 1000.0 / n_crops:.2f} ms/crop end-to-end")


if __name__ == "__main__":
    main()
