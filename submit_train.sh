#!/bin/bash
# submit a WS-DINO training run from scratch to the Janelia LSF cluster.
#
# usage:
#   bsub < submit_train.sh
#
# this script starts training at step 0. once the first checkpoint exists
# (after ckpt_interval steps), use submit_resume.sh to continue from where
# the job left off.

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
#BSUB -J predicty_train              # job name (shows in bjobs)
#BSUB -q gpu_a100                    # queue. verify with `bqueues` if unsure.
#BSUB -gpu "num=1"                   # request 1 GPU
#BSUB -n 8                           # 8 CPU cores: 4 dataloader workers + headroom
#BSUB -W 72:00                       # walltime cap. adjust to queue maximum.
#BSUB -R "rusage[mem=64000]"         # 64 GB RAM
#BSUB -o logs/lsf/train.%J.out       # stdout log (%J = LSF job id)
#BSUB -e logs/lsf/train.%J.err       # stderr log

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

# launch training. tensorboard logs go to logs/, checkpoints to checkpoints/.
python train.py --config configs/train.yaml
