#!/bin/bash
# entry point that launches WS-DINO runs on the LSF cluster.
#
# every run gets its own name. the name namespaces all output, so several runs
# execute at the same time and no run overwrites another run:
#   checkpoints/<run>/   checkpoints + the frozen config snapshot
#   logs/<run>/          tensorboard event files
#   logs/lsf/<run>/      lsf stdout/stderr + the gpu sampler csv
#
# usage:
#   ./launch.sh <run_name>            # fresh run from step 0
#   ./launch.sh <run_name> --resume   # resume from checkpoints/<run>/latest.pt
#
# the workflow for several variants at the same time:
#   edit configs/train.yaml   ->   ./launch.sh temp04
#   edit configs/train.yaml   ->   ./launch.sh temp07
#   ...
# launch.sh snapshots configs/train.yaml into the run folder at submit time. an
# edit to train.yaml for the next run therefore never disturbs a run that you
# already submitted, even if that run is still in the queue.

set -euo pipefail

run_name="${1:-}"
mode="${2:-train}"

if [ -z "${run_name}" ]; then
    echo "usage: ./launch.sh <run_name> [--resume]" >&2
    exit 1
fi

resume=0
if [ "${mode}" = "--resume" ] || [ "${mode}" = "resume" ]; then
    resume=1
elif [ "${mode}" != "train" ]; then
    echo "ERROR: unknown second argument '${mode}'. expected --resume or nothing." >&2
    exit 1
fi

ckpt_dir="checkpoints/${run_name}"
lsf_dir="logs/lsf/${run_name}"
snapshot="${ckpt_dir}/config.yaml"

mkdir -p "${ckpt_dir}" "${lsf_dir}" "logs/${run_name}"

if [ "${resume}" -eq 1 ]; then
    # a resume reuses the config that launch.sh froze at the original launch.
    if [ ! -f "${ckpt_dir}/latest.pt" ]; then
        echo "ERROR: ${ckpt_dir}/latest.pt not found. launch a fresh run first." >&2
        exit 1
    fi
    if [ ! -f "${snapshot}" ]; then
        echo "ERROR: ${snapshot} not found. cannot resume without the frozen config." >&2
        exit 1
    fi
    job_script="submit_resume.sh"
    job_name="${run_name}_resume"
    out="${lsf_dir}/resume.out"
    err="${lsf_dir}/resume.err"
else
    # freeze the current config, so a later edit to train.yaml cannot change the
    # data that this run trains on, even if the run waits in the queue.
    if [ -e "${snapshot}" ]; then
        echo "ERROR: ${snapshot} already exists (run '${run_name}' was already launched)." >&2
        echo "pick a new run name, or pass --resume to continue it." >&2
        exit 1
    fi
    cp configs/train.yaml "${snapshot}"
    job_script="submit_train.sh"
    job_name="${run_name}"
    out="${lsf_dir}/train.out"
    err="${lsf_dir}/train.err"
fi

echo "[launch] run      : ${run_name}  ($([ "${resume}" -eq 1 ] && echo resume || echo fresh))"
echo "[launch] config   : ${snapshot}"
echo "[launch] ckpts    : ${ckpt_dir}/"
echo "[launch] lsf logs : ${out} , ${err}"

# the command-line -J/-oo/-eo override the static directives in the job script.
# -env exports RUN_NAME into the job, and the job derives its config path and
# its checkpoint path from RUN_NAME. "all" also keeps the rest of the
# submission environment.
bsub -J "${job_name}" -oo "${out}" -eo "${err}" \
    -env "all, RUN_NAME=${run_name}" \
    < "${job_script}"
