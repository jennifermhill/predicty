"""dataloader smoke test and throughput benchmark (not a training entry point).

pulls a handful of batches through the same MultiZarrDataset / ZarrCellDataset
pipeline that train.py uses and prints per-crop timing plus batch shapes. use it
to confirm the data path works and to spot data-loading bottlenecks before
committing a GPU to a real run. the actual training entry point is train.py.

usage:
    python benchmark_dataloader.py --config configs/train.yaml
"""

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
    batch_size = 4
    cfg.batch_size = batch_size

    # multi-zarr training takes precedence; single-zarr falls back to the picker.
    if cfg.zarr_paths:
        if cfg.zarr_path is not None:
            print(
                "[benchmark] cfg.zarr_paths is set; ignoring cfg.zarr_path "
                f"({cfg.zarr_path!r})."
            )
        ds: ZarrCellDataset | MultiZarrDataset = MultiZarrDataset(cfg)
    else:
        if cfg.zarr_path is None:
            cfg.zarr_path = pick_zarr_path()
        ds = ZarrCellDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.num_workers > 0,
    )

    per_crop_ms: list[float] = []
    n_batches = 20
    # crops per sample: 2 globals (one at t1, one at t2) + n_locals.
    crops_per_sample = 2 + cfg.n_local_crops
    print(
        f"[benchmark] pulling {n_batches} batches of {batch_size} samples "
        f"({crops_per_sample} crops/sample) ..."
    )

    t_loop = time.time()
    for i, batch in zip(range(n_batches), loader):
        t0 = time.time()
        # globals: (B, 2, 1, Z, H, W); dim 1 indexes (t1, t2).
        globals_ = batch["globals"]
        g_t1 = globals_[:, 0]
        g_t2 = globals_[:, 1]
        locs = batch.get("locals")
        meta = batch["meta"]
        dt = (time.time() - t0) * 1000.0
        per_crop_ms.append(dt / (batch_size * crops_per_sample))
        ds_idx = meta["dataset_idx"].tolist() if "dataset_idx" in meta else None
        locals_shape = tuple(locs.shape) if locs is not None else None
        print(
            f"[benchmark] batch {i:02d} globals_t1={tuple(g_t1.shape)} "
            f"globals_t2={tuple(g_t2.shape)} locals={locals_shape} "
            f"t1={meta['t1'].tolist()} t2={meta['t2'].tolist()} "
            f"z={meta['z'].tolist()} z_end={meta['z_end'].tolist()}"
            + (f" dataset_idx={ds_idx}" if ds_idx is not None else "")
        )

    total = time.time() - t_loop
    if per_crop_ms:
        print(
            f"[benchmark] median ms/crop (post-fetch handling only): "
            f"{median(per_crop_ms):.2f}"
        )
    n_crops = n_batches * batch_size * crops_per_sample
    print(f"[benchmark] total wall time: {total:.2f}s for {n_crops} crops "
          f"-> {total * 1000.0 / n_crops:.2f} ms/crop end-to-end")


if __name__ == "__main__":
    main()
