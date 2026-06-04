#!/bin/bash
# resume a WS-DINO training run from the most recent checkpoint.
#
# usage:
#   cd /groups/sgro/sgrolab/Ankit/predicty
#   bsub < submit_resume.sh
#
# requires checkpoints/latest.pt to exist (saved every ckpt_interval steps
# by train.py). will fail loudly if it doesn't. submit submit_train.sh first
# to start a fresh run.

# ---------------------------------------------------------------------------
# LSF directives (same shape as submit_train.sh)
# ---------------------------------------------------------------------------
#BSUB -J predicty_resume             # job name (shows in bjobs)
#BSUB -q gpu_a100                    # A100 queue: 12 slots/GPU, 40GB ram/slot
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 8                           # 8 slots = 8 cores + 320 GB ram (slot-implied)
#BSUB -W 72:00                       # 72 hour walltime
#BSUB -oo logs/lsf/resume.out        # stdout log (overwritten on each submit; -oo not -o)
#BSUB -eo logs/lsf/resume.err        # stderr log (overwritten on each submit)

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# LSF needs the log directory to exist before it can write into it.
mkdir -p logs/lsf

# cap per-process threading. without these, every numerical library detects
# the full node's core count and spawns that many threads per process. with
# 4 dataloader workers + 1 main process, value of 2 keeps total threads at
# ~10 across the 8 slots we requested.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# node-local scratch for any temp files (matplotlib cache, dask spills, etc.)
# never use /tmp on compute nodes per cluster policy.
export TMPDIR=/scratch/$USER/$LSB_JOBID
mkdir -p $TMPDIR

# activate the conda env. verify the source path with `which conda` on a
# login node if this fails (path may differ from ~/miniconda3 on the cluster).
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# fail with a clear error if there's nothing to resume from.
if [ ! -f checkpoints/latest.pt ]; then
    echo "ERROR: checkpoints/latest.pt not found." >&2
    echo "submit submit_train.sh first to start a fresh run." >&2
    exit 1
fi

# spin up a background nvidia-smi sampler. polls every 30s and writes CSV
# rows to logs/lsf/gpu.out so we can confirm the GPU is actually doing work.
# the EXIT trap kills it whenever this shell exits (normal end, error, bkill).
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    > logs/lsf/gpu.out 2>&1 &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true' EXIT

# resume from the most recent checkpoint. train.py loads student + teacher
# + optimizer + scaler + center + step counter, then continues from
# saved_step + 1 until cfg.total_steps.
python train.py --config configs/train.yaml --resume checkpoints/latest.pt

# clean up node-local scratch on the way out.
rm -rf $TMPDIR
