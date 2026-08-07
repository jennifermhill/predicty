#!/bin/bash
# LSF job body for the evaluation pipeline (latent space viz + dataset/behavior
# probes + grad-cam) over one or more complete training runs.
#
# do not submit this file directly. launch the file through launch_eval.sh.
# launch_eval.sh reads num_workers out of each run's frozen config.yaml to size
# -n, and it overrides -J/-n/-oo/-eo, so concurrent eval jobs stay separate:
#
#   cd /groups/sgro/sgrolab/Aran/predicty      # the checkout location of this repo
#   ./launch_eval.sh <run_name> [<run_name2> ...]
#
# each run must already have checkpoints/<run_name>/config.yaml, which
# ./launch.sh froze, and checkpoints/<run_name>/latest.pt.

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
# launch_eval.sh overrides -J/-n/-oo/-eo per submission. the values here are
# only the fallback for a raw `bsub < submit_eval.sh`.
#BSUB -J predicty_eval                    # job name. bjobs shows this name.
#BSUB -q gpu_a100                         # A100 queue
#BSUB -gpu "num=1"                        # request 1 GPU
#BSUB -n 6                                # fallback slot count. launch_eval.sh sets the count to num_workers+2
#BSUB -W 24:00                            # 24 hour walltime
#BSUB -oo logs/lsf/eval_default/eval.out  # stdout log. -oo, not -o, so LSF overwrites the log on each submit.
#BSUB -eo logs/lsf/eval_default/eval.err  # stderr log. LSF overwrites the log on each submit.

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# launch_eval.sh exports both variables, and a raw bsub falls back to
# 'default'. an un-namespaced submit therefore still works, but a run with the
# literal name 'default' must exist.
EVAL_NAME="${EVAL_NAME:-default}"
RUN_NAMES="${RUN_NAMES:-default}"
lsf_dir="logs/lsf/eval_${EVAL_NAME}"

# LSF needs the log directory before LSF can write into the directory.
mkdir -p "${lsf_dir}"

# cap the thread count per process, for the same reason as submit_train.sh.
# without these variables every numerical library detects the full node's core
# count and spawns that many threads per process.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# warning: never use /tmp on a compute node.
# node-local scratch for any temp file.
export TMPDIR=/scratch/$USER/$LSB_JOBID
mkdir -p $TMPDIR

# activate the conda env. if the activation fails, verify the source path with
# `which conda` on a login node.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# background nvidia-smi sampler, the same pattern as submit_train.sh and
# submit_resume.sh.
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    > "${lsf_dir}/gpu.out" 2>&1 &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true' EXIT

# evaluation_pipeline.py takes each run's config from
# checkpoints/<run>/config.yaml and each run's checkpoint from
# checkpoints/<run>/latest.pt, so the script needs only the run names.
python eval/evaluation_pipeline.py --run-names ${RUN_NAMES}

# clean up the node-local scratch at the end.
rm -rf $TMPDIR
