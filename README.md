# predicty

Self-supervised representation learning for cell-movement behavior in 3D
microscopy videos.

The goal is a single artifact: an **encoder** that maps a high-dimensional cell
crop (a small 3D sub-volume from a microscopy video) to a compact embedding,
trained **without labels** so that crops with similar behavior land near each
other. A small linear classifier is then trained on top of the frozen encoder to
recognize behaviors (streaming, slug movement, clumping, etc.). Everything else
in this repo is scaffolding to train and evaluate that encoder.

The training method is **WS-DINO** (weakly-supervised DINO): a student/teacher
self-supervised setup where the two "views" of a scene are the same field of
view at two nearby timepoints (`t1`, `t2`), plus augmentations. The teacher is
an EMA of the student; centering and sharpening prevent collapse.

---

## Repository layout

```
predicty/
├── README.md                 you are here
├── train.py                  ← training entry point
├── train_utils.py            schedules, checkpoint I/O, device helpers used by train.py
├── benchmark_dataloader.py   dataloader smoke test + throughput benchmark (NOT training)
│
├── configs/
│   ├── config.py             Config dataclass (every knob + its default) and yaml loader
│   └── train.yaml            the config you actually edit for a run
│
├── data/                     the data pipeline (zarr -> augmented crop pairs)
│   ├── dataset.py            ZarrCellDataset / MultiZarrDataset (the core loader)
│   ├── picker.py             interactive/CLI zarr chooser for single-zarr runs
│   ├── metadata.py           reads training_data_index.xlsx (per-experiment quirks)
│   ├── balancing.py          equal per-dataset representation within a batch
│   ├── spatial_regions.py    spatial-region holdout for leakage-controlled probes
│   └── augmentations.py      semantically-preserving crop augmentations
│
├── models/
│   ├── encoder.py            HybridEncoder: 3D stem collapses z, then 2D ResNet-50
│   ├── dino.py               DINO head, student/teacher wrapper, DINO + iBOT losses
│   └── mae.py                masked-patch autoencoder (auxiliary reconstruction loss)
│
├── eval/                     everything for inspecting a trained encoder (see below)
│
├── scripts/
│   └── filter_annotations.py filter index.jsonl to the datasets in train.yaml
│
├── tests/                    pytest/unittest suite for the non-GPU pieces
├── notebooks/visualize.ipynb scratch visualization notebook
│
├── launch.sh                 submit a run to the LSF cluster (wraps the two below)
├── submit_train.sh           LSF job body: fresh run
├── submit_resume.sh          LSF job body: resume from checkpoint
│
├── index.jsonl               behavior/dataset annotations (one JSON object per line)
├── training_data_index.xlsx  per-experiment metadata (frame interval, z limits)
├── environment.yaml          full conda export (Linux; exact pinned build strings)
└── requirements.txt          portable pip requirements
```

---

## Setup

The project targets Python 3.11 with PyTorch and CUDA on a Linux GPU node.

**Option A — conda (matches the training cluster exactly):**

```bash
conda env create -f environment.yaml
conda activate predicty
```

`environment.yaml` is a full `conda env export` with pinned Linux build strings,
so it reproduces the cluster environment but is not portable to macOS/Windows.

**Option B — pip (portable):**

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

A CUDA-capable GPU is required for training; evaluation of a saved checkpoint can
run on CPU (slowly).

---

## Data

- Inputs are **zarr** arrays of microscopy video, indexed `(t, c, z, y, x)`.
  Data is not uniform across experiments; per-experiment quirks (frame interval
  in minutes, usable `z` range) live in `training_data_index.xlsx` and are read
  by `data/metadata.py`. Check that spreadsheet before touching loader code.
- The z axis is lower-resolution than x/y (anisotropic); the loader and crop
  geometry account for this.
- **Pairing:** each training sample is one field of view at two timepoints,
  `t1` and `t2`, spaced `target_gap_minutes` apart (rounded to a whole frame gap
  using the per-experiment interval). This temporal pair is the weak label.
- **Annotations** (`index.jsonl`) are only used for the downstream *probes*, not
  for pre-training. Each line records a crop's location, source zarr, and
  behavior `label`. `scripts/filter_annotations.py` narrows it to the datasets
  the current `configs/train.yaml` trains on.

---

## Training

All settings live in `configs/train.yaml` (documented inline; the authoritative
list of every field and its default is the `Config` dataclass in
`configs/config.py`). Pick **either** `zarr_path` (single dataset) **or**
`zarr_paths` (multi-dataset mixing).

**On the LSF cluster (recommended):**

```bash
./launch.sh my_run             # fresh run from step 0
./launch.sh my_run --resume    # resume from checkpoints/my_run/latest.pt
```

`launch.sh` gives each run a name, **snapshots `configs/train.yaml`** into
`checkpoints/<run>/config.yaml` at submit time (so editing the yaml for the next
run never disturbs an already-queued one), and namespaces all outputs:

```
checkpoints/<run>/   checkpoints + the frozen config snapshot
logs/<run>/          tensorboard event files
logs/lsf/<run>/      LSF stdout/stderr + a periodic nvidia-smi GPU sampler
```

**Directly (local smoke test or non-LSF machine):**

```bash
python train.py --config configs/train.yaml --run-name my_run
python train.py --config configs/train.yaml --device cpu   # tiny CPU smoke test
```

**Monitor:**

```bash
tensorboard --logdir logs
```

Watch `collapse/embedding_var` (should stay well off the floor) and
`dataset_adversary/acc` (should trend toward chance). If variance collapses,
check augmentation strength and learning rate first.

**Before committing a GPU,** sanity-check the data path and its throughput:

```bash
python benchmark_dataloader.py --config configs/train.yaml
```

### What gets trained

The encoder is optimized against a DINO self-distillation loss, plus optional
auxiliary objectives toggled in the config:

- **DINO** (always): student/teacher self-distillation over the crop views.
- **MAE or iBOT** (`mae_enabled` / `ibot_enabled`, mutually exclusive): masked
  reconstruction of global crops through a training-only decoder.
- **Dataset adversary** (`dataset_adversary_enabled`, multi-zarr only): a
  gradient-reversal classifier that pushes the encoder to drop dataset identity
  (batch-effect removal). Keep its pressure weak until DINO is stable.

The projector/decoder/adversary are all discarded after training; only the
encoder ships.

---

## Evaluation (`eval/`)

Run these against a checkpoint after training. Most take
`--config <path> --checkpoint checkpoints/<run>/latest.pt`.

| Script | What it does |
| --- | --- |
| `latent_space.py` | Embed sampled crops, UMAP them, and cache `embeddings.npz` + `metadata.csv` under `latent_space/<run>_*/`. Run this first; several tools reuse its cache. |
| `probe_behavior.py` | Linear probe for behavior identity on the frozen encoder. Should **succeed** (embeddings transfer to the task). |
| `probe_dataset.py` | Linear probe for dataset identity. Should **fail** (encoder learned dataset-invariant features). |
| `wasserstein.py` / `probe_wasserstein.py` | Sliced-Wasserstein batch-effect metric between per-dataset embedding clouds. `wasserstein.py` is the reusable numpy-only module; `probe_wasserstein.py` runs it over a cached `embeddings.npz`. |
| `gradcam.py` + `run_gradcam.py` + `visualize_gradcam.py` | Grad-CAM attribution (library, CLI runner, and plotting, respectively). |
| `expected_gradients.py` + `run_eg.py` | Expected-Gradients input-space attribution (library + CLI runner). |
| `interactive_latent_viewer.py` | Click points in a 2D latent map to see the raw crops behind them. |
| `visualize_mae_mask.py` | Visualize the masked patches the MAE loss reconstructs. |
| `evaluation_pipeline.py` | Orchestrator: run latent-space viz + both probes across several runs, skipping already-evaluated ones. |

Example end-to-end for one run:

```bash
python eval/latent_space.py   --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt
python eval/probe_behavior.py --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt
python eval/probe_dataset.py  --config checkpoints/my_run/config.yaml --checkpoint checkpoints/my_run/latest.pt
# or all of the above for many runs at once:
python eval/evaluation_pipeline.py --config checkpoints/my_run/config.yaml --run-names my_run other_run
```

---

## Tests

```bash
pytest            # or: python -m pytest
```

Tests cover the non-GPU pieces (batching, spatial regions, MAE-mask metrics, the
latent viewer). They still import the training stack, so run them inside the
`predicty` environment.

---

## Things a new maintainer should know

- **Cluster-specific paths.** `configs/train.yaml` (`zarr_paths`,
  `annotation_path`), the `submit_*.sh` conda activation path
  (`~/miniconda3/...`), and the LSF directives (`gpu_h200` queue) are all
  specific to the Janelia/Sgro-lab setup. Point them at your own data, env, and
  scheduler before running elsewhere. `launch.sh` assumes `bsub` (LSF).
- **Run artifacts are gitignored** and never committed: `checkpoints/`, `logs/`,
  `latent_space/`, `outputs/`, `mae_masks/`, `annotations/`, `plans/`. They live
  only on the machine that produced them.
- **`benchmark_dataloader.py` is not a training entry point** despite living at
  the repo root. `train.py` is. The benchmark just pulls a few batches and
  reports per-crop timing.
- **Two dependency files, two purposes.** `environment.yaml` is an exact conda
  export (Linux build strings, not portable); `requirements.txt` is the portable
  pip list. Update both when you change dependencies.
- **Coding conventions** are documented in `CLAUDE.md` (kept locally, not
  committed): lowercase comments, no em dashes, `snake_case` / `PascalCase`,
  hyphenated CLI flags (`--run-name`), and always save both encoder and full
  model state in checkpoints so the encoder can be extracted independently.
- **Don't commit secrets.** Scratch notes and personal tokens do not belong in
  tracked files; `notes.txt` is gitignored for that reason.
