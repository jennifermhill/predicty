"""positive-pair sampler over an ome-multiscale zarr.

a positive pair is two zyx crops at the same anchor but different timepoints.
the time gap, z range, and t1 range are fractions of each dataset by default.

the dataset is iterable because the crop space is too large to index usefully.

the zarr is reopened in each worker so dask handles stay process-local.
"""

from __future__ import annotations

import os
import random
from typing import Any, Iterator

import dask.array as da
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import IterableDataset, get_worker_info

from config import Config
from data.augmentations import build_augmentations


class ZarrCellDataset(IterableDataset):
    def __init__(self, cfg: Config) -> None:
        super().__init__()
        if cfg.zarr_path is None:
            raise ValueError("cfg.zarr_path must be set before constructing ZarrCellDataset")
        self.cfg = cfg

        # open once here to read shape and resolve ranges.
        self._open_zarr()
        self._resolve_ranges()
        self._log_open()

        # augmentations are stateless across workers.
        self._augment = build_augmentations(cfg)

    def _open_zarr(self) -> None:
        cfg = self.cfg
        root = cfg.zarr_path
        candidate = f"{root}/0/{cfg.pyramid_level}"
        if _path_exists(candidate):
            self.arr = da.from_zarr(candidate)
            self.runtime_downsample = cfg.fallback_downsample
            self.effective_level = cfg.pyramid_level
        else:
            self.arr = da.from_zarr(f"{root}/0/0")
            self.runtime_downsample = 2 ** cfg.pyramid_level
            self.effective_level = 0

        # shape varies across datasets.
        # expected layout is (t, c, z, y, x).
        if self.arr.ndim != 5:
            raise ValueError(
                f"Expected 5D zarr (T, C, Z, Y, X); got shape {self.arr.shape}"
            )
        self.T, self.C, self.Z, self.Y, self.X = self.arr.shape
        if self.T < 2:
            raise ValueError(f"need at least 2 timepoints; got T={self.T}")

    def _resolve_ranges(self) -> None:
        cfg = self.cfg

        self._check_fraction("delta_t_frac_min", cfg.delta_t_frac_min)
        self._check_fraction("delta_t_frac_max", cfg.delta_t_frac_max)
        self._check_fraction("z_frac_min", cfg.z_frac_min)
        self._check_fraction("z_frac_max", cfg.z_frac_max)
        self._check_fraction("t1_frac_min", cfg.t1_frac_min)
        self._check_fraction("t1_frac_max", cfg.t1_frac_max)
        self._check_order("delta_t_frac", cfg.delta_t_frac_min, cfg.delta_t_frac_max)
        self._check_order("z_frac", cfg.z_frac_min, cfg.z_frac_max)
        self._check_order("t1_frac", cfg.t1_frac_min, cfg.t1_frac_max)

        if cfg.z_thickness < 1:
            raise ValueError(f"z_thickness must be >= 1; got {cfg.z_thickness}")

        # convert fractional time gaps to frames for this dataset.
        self.delta_t_min = min(
            self.T - 1,
            max(1, int(round(self.T * cfg.delta_t_frac_min))),
        )
        self.delta_t_max = min(
            self.T - 1,
            max(self.delta_t_min, int(round(self.T * cfg.delta_t_frac_max))),
        )

        # absolute z bounds are only for pinning a known dataset.
        if cfg.z_abs_min is not None and cfg.z_abs_max is not None:
            self.z_min = cfg.z_abs_min
            self.z_max = cfg.z_abs_max
        else:
            self.z_min = int(round(self.Z * cfg.z_frac_min))
            self.z_max = max(self.z_min + 1, int(round(self.Z * cfg.z_frac_max)))
        self.z_min = max(0, min(self.z_min, self.Z - 1))
        self.z_max = max(self.z_min + 1, min(self.z_max, self.Z))

        if self.z_max - self.z_min < cfg.z_thickness:
            raise ValueError(
                f"z range [{self.z_min}, {self.z_max}) has "
                f"{self.z_max - self.z_min} slices, but z_thickness={cfg.z_thickness}"
            )

        # absolute t bounds are only for pinning a known dataset.
        if cfg.t1_abs_min is not None and cfg.t1_abs_max is not None:
            self.t1_min = cfg.t1_abs_min
            self.t1_max = cfg.t1_abs_max
        else:
            self.t1_min = int(round(self.T * cfg.t1_frac_min))
            self.t1_max = max(self.t1_min + 1, int(round(self.T * cfg.t1_frac_max)))
        self.t1_min = max(0, min(self.t1_min, self.T - 1))
        self.t1_max = max(self.t1_min + 1, min(self.t1_max, self.T))

    @staticmethod
    def _check_fraction(name: str, value: float) -> None:
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1]; got {value}")

    @staticmethod
    def _check_order(name: str, low: float, high: float) -> None:
        if low > high:
            raise ValueError(f"{name}_min must be <= {name}_max; got {low} > {high}")

    def _log_open(self) -> None:
        print(
            f"[ZarrCellDataset] opened {self.cfg.zarr_path} "
            f"shape={tuple(self.arr.shape)} chunks={self.arr.chunks} "
            f"level={self.effective_level} runtime_downsample={self.runtime_downsample}"
        )
        print(
            f"[ZarrCellDataset] T={self.T}; delta_t resolved to "
            f"[{self.delta_t_min}, {self.delta_t_max}] frames"
        )
        print(
            f"[ZarrCellDataset] Z={self.Z}; z range resolved to "
            f"[{self.z_min}, {self.z_max}) slices, thickness={self.cfg.z_thickness}"
        )
        print(
            f"[ZarrCellDataset] t1 range resolved to "
            f"[{self.t1_min}, {self.t1_max}) frames"
        )

    def __iter__(self) -> Iterator[dict[str, Any]]:
        info = get_worker_info()
        if info is None:
            rng = random.Random(self.cfg.seed)
        else:
            rng = random.Random(self.cfg.seed + info.id)

        while True:
            yield self._sample_pair(rng)

    def _sample_pair(self, rng: random.Random) -> dict[str, Any]:
        cfg = self.cfg
        H = W = cfg.crop_size

        # crop at array scale, then downsample y/x if needed.
        slice_h = H * self.runtime_downsample
        slice_w = W * self.runtime_downsample
        if slice_h > self.Y or slice_w > self.X:
            raise ValueError(
                f"crop_size={cfg.crop_size} with runtime_downsample="
                f"{self.runtime_downsample} requires {slice_h}x{slice_w} pixels, "
                f"but frame is only {self.Y}x{self.X}"
            )

        # sample one z stack and reuse it for both views.
        z_start_max = self.z_max - cfg.z_thickness
        z = rng.randrange(self.z_min, z_start_max + 1)
        t1 = rng.randrange(self.t1_min, self.t1_max)

        # sample a signed time gap.
        delta = rng.randint(self.delta_t_min, self.delta_t_max)
        sign = rng.choice((-1, 1))
        t2 = int(np.clip(t1 + sign * delta, 0, self.T - 1))
        # flip once if clipping landed back on t1.
        if t2 == t1:
            t2 = int(np.clip(t1 - sign * delta, 0, self.T - 1))

        # sample one spatial crop and reuse it for both views.
        y_off = rng.randrange(0, self.Y - slice_h + 1)
        x_off = rng.randrange(0, self.X - slice_w + 1)

        view1 = self._load_crop(t1, z, y_off, x_off, slice_h, slice_w)
        view2 = self._load_crop(t2, z, y_off, x_off, slice_h, slice_w)

        view1 = self._augment(view1)
        view2 = self._augment(view2)

        return {
            "view1": view1,
            "view2": view2,
            "meta": {
                "t1": t1,
                "t2": t2,
                "z": z,
                "z_end": z + cfg.z_thickness,
                "x": x_off,
                "y": y_off,
            },
        }

    def _load_crop(
        self,
        t: int,
        z: int,
        y_off: int,
        x_off: int,
        slice_h: int,
        slice_w: int,
    ) -> torch.Tensor:
        cfg = self.cfg
        # index the smallest useful dask slice.
        sub = self.arr[
            t,
            cfg.channel,
            z : z + cfg.z_thickness,
            y_off : y_off + slice_h,
            x_off : x_off + slice_w,
        ]
        np_arr = np.asarray(sub.compute(), dtype=np.float32)

        # rescale before augmentations.
        np_arr = np.clip(np_arr, 0.0, cfg.max_intensity) / cfg.max_intensity

        # shape is [1, z, h, w].
        crop = torch.from_numpy(np_arr).unsqueeze(0)

        if self.runtime_downsample > 1:
            # downsample y/x while keeping z thickness unchanged.
            crop = F.interpolate(
                crop.unsqueeze(0),
                scale_factor=(
                    1.0,
                    1.0 / self.runtime_downsample,
                    1.0 / self.runtime_downsample,
                ),
                mode="trilinear",
                align_corners=False,
            ).squeeze(0)

        return crop


def worker_init_fn(worker_id: int) -> None:
    """re-open the zarr handle in each worker."""
    info = get_worker_info()
    if info is None:
        return
    ds: ZarrCellDataset = info.dataset  # type: ignore[assignment]
    ds._open_zarr()
    # seed augmentations per worker.
    seed = ds.cfg.seed + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


def _path_exists(path: str) -> bool:
    # this project uses filesystem-backed zarr stores.
    return os.path.isdir(path)
