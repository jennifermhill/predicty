#!/bin/bash
# LSF job body for the evaluation pipeline (latent space viz + dataset/behavior
# probes + grad-cam) over one or more finished training runs.
#
# do not submit this directly. launch it through launch_eval.sh, which reads
# num_workers out of each run's frozen config.yaml to size -n and overrides
# -J/-n/-oo/-eo so concurrent eval jobs stay separate:
#
#   cd /groups/sgro/sgrolab/Ankit/predicty
#   ./launch_eval.sh <run_name> [<run_name2> ...]
#
# each run must already have checkpoints/<run_name>/config.yaml (frozen by
# ./launch.sh) and checkpoints/<run_name>/latest.pt.

# ---------------------------------------------------------------------------
# LSF directives
# ---------------------------------------------------------------------------
# -J/-n/-oo/-eo are overridden per-submission by launch_eval.sh; the values
# here are only the fallback for a raw `bsub < submit_eval.sh`.
#BSUB -J predicty_eval                    # job name (shows in bjobs)
#BSUB -q gpu_a100                         # A100 queue
#BSUB -gpu "num=1"                        # request 1 GPU
#BSUB -n 6                                # fallback slot count; launch_eval.sh sets it to num_workers+2
#BSUB -W 24:00                            # 24 hour walltime
#BSUB -oo logs/lsf/eval_default/eval.out  # stdout log (overwritten on each submit; -oo not -o)
#BSUB -eo logs/lsf/eval_default/eval.err  # stderr log (overwritten on each submit)

# ---------------------------------------------------------------------------
# script body
# ---------------------------------------------------------------------------
set -euo pipefail

# launch_eval.sh exports both of these; a raw bsub falls back to 'default'
# so an un-namespaced submit still works (but requires a run literally named
# 'default' to exist).
EVAL_NAME="${EVAL_NAME:-default}"
RUN_NAMES="${RUN_NAMES:-default}"
lsf_dir="logs/lsf/eval_${EVAL_NAME}"

# LSF needs the log directory to exist before it can write into it.
mkdir -p "${lsf_dir}"

# cap per-process threading, same rationale as submit_train.sh: without this
# every numerical library detects the full node's core count and spawns that
# many threads per process.
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# node-local scratch for any temp files. never use /tmp on compute nodes.
export TMPDIR=/scratch/$USER/$LSB_JOBID
mkdir -p $TMPDIR

# activate the conda env. verify the source path with `which conda` on a
# login node if this fails.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate predicty

# background nvidia-smi sampler, same pattern as submit_train.sh/submit_resume.sh.
nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw \
    --format=csv \
    --loop=30 \
    > "${lsf_dir}/gpu.out" 2>&1 &
GPU_SAMPLER_PID=$!
trap 'kill ${GPU_SAMPLER_PID} 2>/dev/null || true' EXIT

# evaluation_pipeline.py derives each run's config from
# checkpoints/<run>/config.yaml and checkpoint from checkpoints/<run>/latest.pt,
# so it only needs the run names.
python eval/evaluation_pipeline.py --run_names ${RUN_NAMES}

# clean up node-local scratch on the way out.
rm -rf $TMPDIR
