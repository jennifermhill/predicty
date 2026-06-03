#!/bin/bash
# submit a WS-DINO training run from scratch to the Janelia LSF cluster.
#
# usage:
#   cd /groups/sgro/sgrolab/Ankit/predicty
#   bsub < submit_train.sh
#
# this script starts training at step 0. once the first checkpoint exists
# (after ckpt_interval steps), use submit_resume.sh to continue from where
# the job left off.

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
#BSUB -J predicty_train              # job name (shows in bjobs)
#BSUB -q gpu_a100                    # A100 queue: 12 slots/GPU, 40GB ram/slot
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 8                           # 8 slots = 8 cores + 320 GB ram (slot-implied)
#BSUB -W 72:00                       # 72 hour walltime
#BSUB -oo logs/lsf/train.out         # stdout log (overwritten on each submit; -oo not -o)
#BSUB -eo logs/lsf/train.err         # stderr log (overwritten on each submit)

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

# launch training. tensorboard logs go to logs/, checkpoints to checkpoints/.
python train.py --config configs/train.yaml

# clean up node-local scratch on the way out.
rm -rf $TMPDIR
