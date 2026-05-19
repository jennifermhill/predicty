from __future__ import annotations

import time
from statistics import median

from torch.utils.data import DataLoader

from config import Config
from data.dataset import ZarrCellDataset, worker_init_fn
from data.picker import pick_zarr_path


def main() -> None:
    cfg = Config()
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
        print(
            f"[main] batch {i:02d} view1={tuple(v1.shape)} view2={tuple(v2.shape)} "
            f"t1={meta['t1'].tolist()} t2={meta['t2'].tolist()} "
            f"z={meta['z'].tolist()} z_end={meta['z_end'].tolist()}"
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
