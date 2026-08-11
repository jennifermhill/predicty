"""post-hoc latent decoder. the training happens after the fact, on a frozen encoder.

no current checkpoint can decode a latent to voxels. iBOT's masked-prediction
head (models/dino.py, `masking_head`) distills in an abstract projector_k-dim
space with no pixel decoder at all. train.py force-disables MAE's decoder
(models/ibot.py, `MaskedPatchAutoencoder`) whenever ibot_enabled is set, so its
weights never trained. this module supplies the missing latent -> voxel mapping.
the module trains a decoder against a frozen encoder, whatever masking objective
the encoder itself used. a trained decoder is what makes
`interpolate_and_decode` possible: the code walks a straight line between two
real embeddings and decodes the intermediate points, which match no real crop.

the decoder is deliberately a plain autoencoder, not a re-implementation of MAE.
there is no masking and no mask token. the decoder therefore behaves on
arbitrary latent vectors. a masked autoencoder instead specializes in the
prediction of missing patches.

the code caches the crops and the embeddings up front, so that the training
never re-reads zarr. the code then splits them by the same spatial-holdout rule
that probe_dataset.py uses (see "How to read the metrics" in README.md). see
--n-samples and --val-frac.

usage:
    python eval/decoder.py --config configs/train.yaml --checkpoint checkpoints/final/latest.pt --steps 5000
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

# this line lets you run `python eval/decoder.py` from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs.config import Config
from data.dataset import MultiZarrDataset, worker_init_fn
from data.spatial_regions import region_bounds_for_top_left
from models.encoder import HybridEncoder
from models.ibot import _reconstruction_metrics, _trunc_normal_


# ---------------------------------------------------------------------------
# patchify / unpatchify -- MaskedPatchAutoencoder's reshape and permute logic as
# free functions, because LatentDecoder has no mask machinery to hold them.
# ---------------------------------------------------------------------------


def patchify(
    x: torch.Tensor, crop_shape: tuple[int, int, int], patch_shape: tuple[int, int, int]
) -> torch.Tensor:
    """(N, 1, Z, H, W) -> (N, n_patches, patch_dim)."""
    n, c, z, h, w = x.shape
    pz, ph, pw = patch_shape
    if c != 1 or (z, h, w) != tuple(crop_shape):
        raise ValueError(f"expected crop shape (N, 1, {tuple(crop_shape)}); got {tuple(x.shape)}")
    gz, gh, gw = z // pz, h // ph, w // pw
    n_patches = gz * gh * gw
    patch_dim = pz * ph * pw
    x = x.reshape(n, c, gz, pz, gh, ph, gw, pw)
    x = x.permute(0, 2, 4, 6, 1, 3, 5, 7)
    return x.reshape(n, n_patches, patch_dim)


def unpatchify(
    patches: torch.Tensor, crop_shape: tuple[int, int, int], patch_shape: tuple[int, int, int]
) -> torch.Tensor:
    """(N, n_patches, patch_dim) -> (N, 1, Z, H, W)."""
    n = patches.size(0)
    pz, ph, pw = patch_shape
    z, h, w = crop_shape
    gz, gh, gw = z // pz, h // ph, w // pw
    patches = patches.reshape(n, gz, gh, gw, 1, pz, ph, pw)
    patches = patches.permute(0, 4, 1, 5, 2, 6, 3, 7)
    return patches.reshape(n, 1, z, h, w)


class LatentDecoder(nn.Module):
    """the decoder maps a single global encoder latent back to a full voxel volume.

    the decoder is the decoder half of MaskedPatchAutoencoder without the
    masking machinery. the decoder predicts every patch from the same broadcast
    latent plus that patch's own learned position embedding.
    """

    def __init__(
        self,
        latent_dim: int,
        crop_shape: tuple[int, int, int],
        patch_shape: tuple[int, int, int],
        hidden_dim: int = 512,
        depth: int = 2,
        norm_pix_loss: bool = False,
    ) -> None:
        super().__init__()
        if depth < 1:
            raise ValueError(f"depth must be >= 1; got {depth}")

        self.crop_shape = tuple(int(v) for v in crop_shape)
        self.patch_shape = tuple(int(v) for v in patch_shape)
        self.norm_pix_loss = bool(norm_pix_loss)

        z, h, w = self.crop_shape
        pz, ph, pw = self.patch_shape
        if z % pz or h % ph or w % pw:
            raise ValueError(
                "crop_shape must be divisible by patch_shape; "
                f"got crop_shape={self.crop_shape}, patch_shape={self.patch_shape}"
            )
        self.grid_shape = (z // pz, h // ph, w // pw)
        self.n_patches = math.prod(self.grid_shape)
        self.patch_dim = pz * ph * pw

        self.latent_proj = nn.Linear(latent_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.empty(1, self.n_patches, hidden_dim))

        layers: list[nn.Module] = []
        for _ in range(depth):
            layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU()])
        layers.extend([nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, self.patch_dim)])
        self.decoder = nn.Sequential(*layers)

        _trunc_normal_(self.pos_embed)

    def _predict_patches(self, z: torch.Tensor) -> torch.Tensor:
        h = self.latent_proj(z).unsqueeze(1)  # (N, 1, hidden_dim)
        tokens = h + self.pos_embed  # (N, n_patches, hidden_dim), broadcasts over N
        return self.decoder(tokens)  # (N, n_patches, patch_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (N, latent_dim) -> reconstructed volume (N, 1, Z, H, W).

        z does not need to come from a real crop. any (N, latent_dim) tensor
        works, and interpolate_and_decode relies on that.
        """
        return unpatchify(self._predict_patches(z), self.crop_shape, self.patch_shape)

    def reconstruction_loss(self, z: torch.Tensor, target_volume: torch.Tensor) -> dict[str, torch.Tensor]:
        """training-time loss: the method compares the patches that z predicts
        against the patches of the target.

        the norm_pix_loss and the metrics are the same as in MAE's masked loss.
        the method uses every patch, and not only the masked patches.
        """
        pred = self._predict_patches(z)
        target = patchify(target_volume, self.crop_shape, self.patch_shape)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True, unbiased=False)
            target = (target - mean) / (var + 1e-6).sqrt()

        loss = F.mse_loss(pred, target)
        with torch.no_grad():
            r2, corr, rel_l2 = _reconstruction_metrics(pred.flatten(), target.flatten())
        return {"loss": loss, "recon_r2": r2, "recon_corr": corr, "recon_rel_l2": rel_l2}


@torch.no_grad()
def interpolate_and_decode(
    decoder: LatentDecoder,
    z_a: torch.Tensor,
    z_b: torch.Tensor,
    n_steps: int = 8,
) -> np.ndarray:
    """the function interpolates linearly between two latent vectors and decodes every point. """
    decoder.eval()
    alphas = torch.linspace(0.0, 1.0, n_steps, device=z_a.device).unsqueeze(1)  # (n_steps, 1)
    z_path = (1 - alphas) * z_a.unsqueeze(0) + alphas * z_b.unsqueeze(0)  # (n_steps, latent_dim)
    volumes = decoder(z_path)  # (n_steps, 1, Z, H, W)
    return volumes.squeeze(1).cpu().numpy()


# ---------------------------------------------------------------------------
# training loop + CLI
# ---------------------------------------------------------------------------

def disable_augmentations(cfg: Config) -> None:
    """the function turns off the augs, so that each latent has one deterministic target crop.

    the frozen encoder is invariant to these augs, and flip and rotation are
    among them. an enabled aug therefore maps one latent to many targets, and
    the decoder learns a blurred average.
    """
    for name in ("aug_intensity", "aug_gamma", "aug_noise", "aug_blur", "aug_flip", "aug_rotation"):
        getattr(cfg, name).enabled = False
    # aug_percrop_norm stays enabled. aug_percrop_norm is the input convention
    # that the encoder learned.


def spatial_holdout_split(
    crops: torch.Tensor,
    embeddings: torch.Tensor,
    rows: list[dict[str, Any]],
    cfg: Config,
    val_frac: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """the function splits the cached crops and embeddings into train and val by
    spatial coordinate.

    the geometric rule is the same as in eval/probe_dataset.py's
    spatial_holdout_split
    """
    if not 0.0 < val_frac < 1.0:
        raise ValueError(f"--val-frac must be in (0, 1); got {val_frac}")

    train_idx: list[int] = []
    val_idx: list[int] = []
    for i, row in enumerate(rows):
        y0, y1, x0, x1 = region_bounds_for_top_left(
            int(row["frame_y"]),
            int(row["frame_x"]),
            cfg.global_crop_size,
            cfg.spatial_region_grid_y,
            cfg.spatial_region_grid_x,
            int(row["spatial_region_idx"]),
        )
        split_on_y = (y1 - y0) >= (x1 - x0)
        lo = y0 if split_on_y else x0
        hi = y1 if split_on_y else x1
        coord = int(row["global_y"] if split_on_y else row["global_x"])
        threshold = lo + max(1, int(round((hi - lo) * (1.0 - val_frac))))
        (val_idx if coord >= threshold else train_idx).append(i)

    if not train_idx or not val_idx:
        raise ValueError(
            "spatial-holdout split produced an empty train or val set; "
            "increase --n-samples or adjust --val-frac"
        )
    return (
        crops[train_idx],
        embeddings[train_idx],
        crops[val_idx],
        embeddings[val_idx],
    )


@torch.no_grad()
def extract_cache(
    cfg: Config,
    encoder: nn.Module,
    device: torch.device,
    n_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """the function embeds n_samples t1 global crops one time. the function
    caches the crops and the embeddings, so that the decoder trains from memory
    and does not re-read zarr at every step."""
    cfg.seed = seed
    disable_augmentations(cfg)

    if not cfg.zarr_paths:
        raise ValueError("decoder training expects cfg.zarr_paths (a multi-zarr dataset)")
    ds = MultiZarrDataset(cfg)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        worker_init_fn=worker_init_fn,
        pin_memory=device.type == "cuda",
        # the context is spawn and not fork, so that the workers start fresh. a
        # fork after the parent initializes cuda and the zarr/blosc threadpool
        # deadlocks at the first read.
        multiprocessing_context="spawn" if cfg.num_workers > 0 else None,
    )

    encoder.eval()
    crops: list[torch.Tensor] = []
    embeddings: list[torch.Tensor] = []
    rows: list[dict[str, Any]] = []
    seen = 0
    t0 = time.time()
    print_every = max(1, n_samples // (cfg.batch_size * 10))
    iters = 0

    for batch in loader:
        # globals shape: (B, 2, 1, Z, H, W). index 0 is the t1 view.
        g_t1 = batch["globals"][:, 0].to(device, non_blocking=True)
        meta = batch["meta"]

        emb = encoder(g_t1)  # (B, latent_dim)
        crops.append(g_t1.half().cpu())
        embeddings.append(emb.float().cpu())

        for i in range(g_t1.size(0)):
            rows.append({
                "dataset_idx": int(batch["weak_label"][i, 0]),
                "t": int(meta["t1"][i]),
                "z": int(meta["z"][i]),
                "global_y": int(meta["global_y"][i]),
                "global_x": int(meta["global_x"][i]),
                "frame_y": int(meta["frame_y"][i]),
                "frame_x": int(meta["frame_x"][i]),
                "spatial_region_idx": int(meta["spatial_region_idx"][i]),
            })

        seen += g_t1.size(0)
        iters += 1
        if iters % print_every == 0:
            rate = seen / max(time.time() - t0, 1e-6)
            print(f"  [extract] {seen}/{n_samples} crops  ({rate:.1f} crops/s)")
        if seen >= n_samples:
            break

    crops_cache = torch.cat(crops, dim=0)[:n_samples]
    emb_cache = torch.cat(embeddings, dim=0)[:n_samples]
    return crops_cache, emb_cache, rows[:n_samples]


def train_decoder(
    decoder: LatentDecoder,
    crops_train: torch.Tensor,
    z_train: torch.Tensor,
    crops_val: torch.Tensor,
    z_val: torch.Tensor,
    device: torch.device,
    steps: int,
    lr: float,
    batch_size: int,
    log_every: int,
) -> None:
    """the function trains `decoder` over the cached (crop, embedding) pairs.

    the loop accesses neither the encoder nor zarr. the loop mirrors the cached
    training loop in eval/probe_dataset.py.
    """
    decoder.train()
    opt = torch.optim.AdamW(decoder.parameters(), lr=lr)

    # the embeddings are small, so they stay on the device. the crops (float16)
    # stay on the host, and each step moves one minibatch over.
    z_train = z_train.to(device)
    val_z = z_val[:batch_size].to(device)
    val_crops = crops_val[:batch_size].to(device).float()

    n = crops_train.size(0)
    gen = torch.Generator(device=device).manual_seed(0)
    t0 = time.time()

    for step in range(1, steps + 1):
        idx = torch.randint(0, n, (batch_size,), device=device, generator=gen)
        z = z_train[idx]
        x = crops_train[idx.cpu()].to(device).float()

        out = decoder.reconstruction_loss(z, x)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()

        if step == 1 or step % log_every == 0 or step == steps:
            decoder.eval()
            with torch.no_grad():
                val_out = decoder.reconstruction_loss(val_z, val_crops)
            decoder.train()
            rate = step / max(time.time() - t0, 1e-6)
            print(
                f"  [decoder] step {step:5d}/{steps}  loss={out['loss'].item():.4f}  "
                f"r2={out['recon_r2'].item():.3f}  corr={out['recon_corr'].item():.3f}  "
                f"({rate:.2f} steps/s)  |  "
                f"val_loss={val_out['loss'].item():.4f} val_r2={val_out['recon_r2'].item():.3f}"
            )


def setup_logging(checkpoint_path: str) -> Path:
    """the function derives a log directory from the name of the checkpoint's
    parent folder. the function then tees all stdout to the terminal and to a
    log file in that directory."""
    run_name = Path(checkpoint_path).parent.name
    candidates = [p for p in Path("latent_space").glob(f"{run_name}_*") if p.is_dir()]
    if candidates:
        log_dir = max(candidates, key=lambda p: p.name)
    else:
        log_dir = Path("latent_space") / f"{run_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        log_dir.mkdir(parents=True, exist_ok=True)

    log_path = log_dir / f"decoder_{time.strftime('%Y%m%d_%H%M%S')}.log"
    log_file = open(log_path, "a")

    class Tee:
        def write(self, msg):
            sys.__stdout__.write(msg)
            log_file.write(msg)
            log_file.flush()

        def flush(self):
            sys.__stdout__.flush()
            log_file.flush()

    sys.stdout = Tee()
    return log_path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="path to a yaml config (same one used for training)")
    p.add_argument("--checkpoint", required=True, help="path to a *.pt file (latest.pt, final.pt, step_*.pt)")
    p.add_argument("--device", default=None, help="override cfg.device (e.g. 'cpu')")
    p.add_argument(
        "--n-samples", type=int, default=8192,
        help="number of t1 global crops (with embeddings) to cache before training the decoder",
    )
    p.add_argument("--steps", type=int, default=15000, help="optimizer steps for the decoder")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--hidden-dim", type=int, default=512, help="decoder MLP width (matches mae_decoder_hidden default)")
    p.add_argument("--depth", type=int, default=2, help="decoder MLP depth (matches mae_decoder_depth default)")
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument(
        "--spatial-region-mode",
        choices=("config", "all", "exclude_holdout", "holdout"),
        default="holdout",
        help="region(s) to train the decoder on; defaults to the held-out region, same convention as the probes",
    )
    p.add_argument(
        "--val-frac",
        type=float,
        default=0.2,
        help=(
            "fraction of each spatial region's longer axis reserved as a "
            "held-out validation strip -- mirrors probe_dataset.py's "
            "spatial-holdout split so validation crops can't overlap or sit "
            "adjacent to training crops"
        ),
    )
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--demo-interp-steps", type=int, default=8, help="frames in the demo interpolation panel")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    log_path = setup_logging(args.checkpoint)
    print(f"[decoder] logging to  : {log_path}")

    cfg = Config.from_yaml(args.config)
    if args.spatial_region_mode != "config":
        cfg.spatial_region_mode = args.spatial_region_mode
    cfg.seed = args.seed

    device = torch.device(args.device or cfg.device)
    print(f"[decoder] device      : {device}")
    if device.type == "cuda":
        idx = device.index if device.index is not None else 0
        print(f"[decoder] cuda gpu    : {torch.cuda.get_device_name(idx)}")
    print(f"[decoder] config      : {args.config}")
    print(f"[decoder] checkpoint  : {args.checkpoint}")
    print(f"[decoder] n_samples   : {args.n_samples}")
    print(f"[decoder] steps       : {args.steps}")
    print(f"[decoder] spatial mode: {cfg.spatial_region_mode}")
    print()

    cfg.n_local_crops = 0  # the decoder training uses only the global crop, not the local crops

    # --- load the frozen encoder, the same pattern as every other eval/probe_*.py ---
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    print(f"[decoder] checkpoint saved at training step {ckpt.get('step', 'unknown')}")
    encoder = HybridEncoder(
        stem_channels=cfg.stem_channels,
        latent_dim=cfg.latent_dim,
        norm_groups=cfg.norm_groups,
    ).to(device)
    encoder.load_state_dict(ckpt["encoder_state_dict"])
    for p in encoder.parameters():
        p.requires_grad = False
    encoder.eval()

    crop_shape = (cfg.z_thickness, cfg.global_crop_size, cfg.global_crop_size)
    patch_shape = (cfg.mae_patch_z, cfg.mae_patch_size, cfg.mae_patch_size)
    decoder = LatentDecoder(
        latent_dim=cfg.latent_dim,
        crop_shape=crop_shape,
        patch_shape=patch_shape,
        hidden_dim=args.hidden_dim,
        depth=args.depth,
    ).to(device)
    n_params = sum(p.numel() for p in decoder.parameters())
    print(f"[decoder] LatentDecoder: crop_shape={crop_shape} patch_shape={patch_shape} params={n_params:,}")
    print()

    # --- cache: one embedding pass per crop, so the training never re-reads zarr ---
    print(f"[decoder] caching {args.n_samples} crops + embeddings (one encoder pass, no augmentations)...")
    t0 = time.time()
    crops, embeddings, rows = extract_cache(cfg, encoder, device, args.n_samples, args.seed)
    print(
        f"[decoder] cached in {time.time() - t0:.1f}s.  "
        f"crops={tuple(crops.shape)}  embeddings={tuple(embeddings.shape)}"
    )
    crops_train, z_train, crops_val, z_val = spatial_holdout_split(
        crops, embeddings, rows, cfg, args.val_frac
    )
    print(f"[decoder] train n={crops_train.size(0)}  val n={crops_val.size(0)}")
    print()

    # --- train the decoder over the cache. the loop uses neither the encoder nor zarr ---
    print(f"[decoder] training for {args.steps} steps (batch={cfg.batch_size})...")
    t0 = time.time()
    train_decoder(
        decoder,
        crops_train,
        z_train,
        crops_val,
        z_val,
        device,
        steps=args.steps,
        lr=args.lr,
        batch_size=cfg.batch_size,
        log_every=args.log_every,
    )
    print(f"[decoder] done in {time.time() - t0:.1f}s")
    print()

    # --- save ---
    log_dir = Path(log_path).parent
    ckpt_out = {
        "decoder_state_dict": decoder.cpu().state_dict(),
        "latent_dim": cfg.latent_dim,
        "crop_shape": crop_shape,
        "patch_shape": patch_shape,
        "hidden_dim": args.hidden_dim,
        "depth": args.depth,
        "source_checkpoint": args.checkpoint,
        "args": vars(args),
    }
    decoder_path = log_dir / "latent_decoder.pt"
    torch.save(ckpt_out, decoder_path)
    print(f"[decoder] saved decoder to {decoder_path}")
    decoder.to(device)

    # --- demo: the code decodes a path between two real crops. the ground truth
    # sits at both ends, so that the reconstruction fidelity and the morph are
    # both visible ---
    print()
    if crops_train.size(0) < 2:
        print("[decoder] cache too small for a demo pair; skipping")
        return

    # cheap O(N) two-step diameter heuristic. the heuristic finds an
    # approximately farthest-apart pair, so that the interpolation traverses
    # something and does not connect two near identical crops.
    d0 = torch.cdist(z_train[:1], z_train).squeeze(0)
    idx_a = int(d0.argmax())
    d_a = torch.cdist(z_train[idx_a : idx_a + 1], z_train).squeeze(0)
    idx_b = int(d_a.argmax())
    print(
        f"[decoder] demo: interpolating between real crops {idx_a} and {idx_b} "
        f"(latent distance {float(d_a[idx_b]):.2f}, {args.demo_interp_steps} steps)..."
    )

    z_a = z_train[idx_a].to(device)
    z_b = z_train[idx_b].to(device)
    volumes = interpolate_and_decode(decoder, z_a, z_b, n_steps=args.demo_interp_steps)
    # z-mean projection of each crop. the display of the decoded volumes uses
    # the same projection.
    real_a = crops_train[idx_a, 0].float().numpy().mean(axis=0)
    real_b = crops_train[idx_b, 0].float().numpy().mean(axis=0)

    # one row: [real A] [decoded a=0 .. a=1] [real B]. decode(z_a) therefore
    # lands next to real A, for a direct comparison of the reconstruction.
    panels = [real_a] + [volumes[i].mean(axis=0) for i in range(args.demo_interp_steps)] + [real_b]
    titles = (
        ["real A"]
        + [f"a={i / (args.demo_interp_steps - 1):.2f}" for i in range(args.demo_interp_steps)]
        + ["real B"]
    )

    fig, axes = plt.subplots(1, len(panels), figsize=(2.0 * len(panels), 2.4))
    for ax, img, title in zip(axes, panels, titles):
        ax.imshow(img, cmap="gray")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("decoded latent interpolation: real crop A -> real crop B (ends are ground truth)", fontsize=10)
    fig.tight_layout()
    demo_path = log_dir / f"decoder_interp_demo_{time.strftime('%Y%m%d_%H%M%S')}.png"
    fig.savefig(demo_path, dpi=150)
    plt.close(fig)
    print(f"[decoder] saved interpolation demo to {demo_path}")


if __name__ == "__main__":
    main()
