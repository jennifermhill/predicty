from dataclasses import dataclass, field


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
    zarr_path: str | None = None
    channel: int = 0
    pyramid_level: int = 1
    fallback_downsample: int = 1
    crop_size: int = 128
    z_thickness: int = 10
    n_views: int = 2

    # time gap as a fraction of each dataset.
    delta_t_frac_min: float = 0.0025
    delta_t_frac_max: float = 0.025

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
