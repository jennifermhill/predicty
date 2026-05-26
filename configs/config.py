from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, get_type_hints


@dataclass
class AugCfg:
    enabled: bool = True
    p: float = 0.5
    # shared fields keep the aug config compact.
    strength: float = 0.0
    gamma_range: tuple[float, float] = (1.0, 1.0)
    sigma_range: tuple[float, float] = (0.0, 0.0)
    kernel_size: int = 5


@dataclass
class Config:
    # data
    # single-zarr training. set this OR `zarr_paths`, not both.
    # left null so the file picker / --zarr-path flag can fill it at runtime.
    zarr_path: str | None = None

    # multi-zarr training. when non-null, the dataloader mixes pairs from
    # every listed zarr in each batch and `zarr_path` is ignored.
    zarr_paths: list[str] | None = None

    channel: int = 0
    z_thickness: int = 10

    # multi-crop architecture: 2 globals + n locals, where each local is
    # spatially nested inside one of the globals. only globals carry a
    # (t1, t2) pair; locals are single-time at t1.
    n_global_crops: int = 2
    n_local_crops: int = 6
    global_crop_size: int = 256
    local_crop_size: int = 128

    # accept a global crop only if its mean is at least this fraction of the
    # whole-frame mean at t1. lower values are more permissive.
    intensity_min_frac: float = 0.5

    # spreadsheet that maps experiment id -> interval_minutes.
    training_data_index_path: str = "training_data_index.xlsx"

    # desired wall-clock gap between paired frames, in minutes.
    # the dataset converts this to a frame_gap using the experiment's interval.
    # default 50 -> 5 frames at a 10-minute interval, the WS-DINO weak label.
    target_gap_minutes: float = 50.0

    # "forward" -> t2 = t1 + frame_gap; "backward" -> t2 = t1 - frame_gap.
    temporal_direction: str = "forward"

    # z window as a fraction of each dataset.
    z_frac_min: float = 0.0
    z_frac_max: float = 1.0

    # absolute z override for one known dataset.
    z_abs_min: int | None = None
    z_abs_max: int | None = None

    # t1 window as a fraction of each dataset.
    t1_frac_min: float = 0.0
    t1_frac_max: float = 1.0

    # absolute t1 override for one known dataset.
    t1_abs_min: int | None = None
    t1_abs_max: int | None = None

    # augmentations
    aug_intensity: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.8, strength=0.4,
    ))
    aug_gamma: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.8, gamma_range=(0.5, 2.0),
    ))
    aug_noise: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.5, sigma_range=(0.0, 0.05),
    ))
    aug_blur: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=0.5, sigma_range=(0.1, 2.0), kernel_size=5,
    ))
    aug_percrop_norm: AugCfg = field(default_factory=lambda: AugCfg(
        enabled=True, p=1.0,
    ))

    # runtime
    max_intensity: float = 10000.0
    num_workers: int = 4
    seed: int = 0

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """build a Config from a yaml file.

        unspecified keys keep their python defaults so this file does not
        need to enumerate every knob.
        """
        import yaml

        with open(path, "r") as f:
            raw = yaml.safe_load(f) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"top-level yaml in {path} must be a mapping")
        return _build_dataclass(cls, raw)

    def to_yaml(self, path: str | Path) -> None:
        """serialize this config to yaml. tuples are written as lists."""
        import yaml

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(_to_plain(self), f, sort_keys=False)


def _to_plain(obj: Any) -> Any:
    """recursively turn dataclasses/tuples into yaml-safe primitives."""
    if is_dataclass(obj):
        return {k: _to_plain(v) for k, v in asdict(obj).items()}
    if isinstance(obj, tuple):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, list):
        return [_to_plain(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_plain(v) for k, v in obj.items()}
    return obj


def _build_dataclass(cls: type, data: dict[str, Any]) -> Any:
    """instantiate a dataclass from a plain dict, recursing into nested dataclasses."""
    field_names = {f.name for f in fields(cls)}
    unknown = set(data) - field_names
    if unknown:
        raise ValueError(f"unknown keys for {cls.__name__}: {sorted(unknown)}")
    # resolve string annotations (from `from __future__ import annotations`) to real types.
    hints = get_type_hints(cls)
    kwargs: dict[str, Any] = {}
    for name, value in data.items():
        ftype = hints.get(name)
        if isinstance(value, dict) and is_dataclass(ftype):
            kwargs[name] = _build_dataclass(ftype, value)
        else:
            kwargs[name] = value
    return cls(**kwargs)
