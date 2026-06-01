#!/bin/bash
# resume a WS-DINO training run from the most recent checkpoint.
#
# usage:
#   bsub < submit_resume.sh
#
# requires checkpoints/latest.pt to exist (saved every ckpt_interval steps
# by train.py). will fail loudly if it doesn't. submit submit_train.sh first
# to start a run from scratch.

# ---------------------------------------------------------------------------
# LSF directives (same as submit_train.sh)
# ---------------------------------------------------------------------------
#BSUB -J predicty_resume             # job name (shows in bjobs)
#BSUB -q gpu_a100                    # queue. verify with `bqueues` if unsure.
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 8                           # 8 CPU cores: 4 dataloader workers + headroom
#BSUB -W 72:00                       # walltime cap. adjust to queue maximum.
#BSUB -R "rusage[mem=64000]"         # 64 GB RAM
#BSUB -o logs/lsf/resume.%J.out      # stdout log (%J = LSF job id)
#BSUB -e logs/lsf/resume.%J.err      # stderr log

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# LSF needs the log directory to exist before the job starts.
mkdir -p logs/lsf

# activate the conda env. update the source path if conda is installed
# somewhere other than ~/miniconda3 on the cluster (try `which conda`).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# move to the project root. update this path to wherever the repo lives
# on the cluster filesystem.
cd ~/predicty

# fail with a clear error if there's nothing to resume from.
if [ ! -f checkpoints/latest.pt ]; then
    echo "ERROR: checkpoints/latest.pt not found." >&2
    echo "submit submit_train.sh first to start a fresh run." >&2
    exit 1
fi

# resume from the most recent checkpoint. train.py loads student + teacher
# + optimizer + scaler + center + step counter, then continues from
# saved_step + 1 until cfg.total_steps.
python train.py --config configs/train.yaml --resume checkpoints/latest.pt
