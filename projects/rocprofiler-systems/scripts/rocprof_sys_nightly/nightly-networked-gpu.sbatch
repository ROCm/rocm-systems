#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Single-phase nightly run for a GPU node that can reach the network.
# Download, extract, clone, build, and test all happen inside this allocation.
# --variant auto reads the GPU from the KFD topology on that node.
#
# Launch from a login node so the path to the Python launcher is recorded
# before Slurm copies this file into its spool:
#
#   WORK_DIR=/shared/$USER/rocprof-sys-nightly \
#     bash nightly-networked-gpu.sbatch
#
# WORK_DIR must not be world-writable. Edit the #SBATCH block for the site
# GPU queue before launching.
#
# Optional environment:
#   TIER=quick|standard|comprehensive|full   (default: standard)
#   VARIANT=auto                             (default: auto)
#   SOURCE_BUILD=1                           also pass --build-from-source
#   RUN_LABELS=mpi                           re-enable labels excluded from the tier

#SBATCH --job-name=rocprof-sys-nightly
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --partition=gpu

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${NIGHTLY_LAUNCHER:-${HERE}/../run-nightly-tarball-tests.py}"
WORK_DIR="${WORK_DIR:-${SCRATCH:-${HOME}}/rocprof-sys-nightly}"
TIER="${TIER:-standard}"
VARIANT="${VARIANT:-auto}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ ! -f "${LAUNCHER}" ]]; then
        echo "Cannot find ${LAUNCHER}. Set NIGHTLY_LAUNCHER to run-nightly-tarball-tests.py." >&2
        exit 1
    fi
    mkdir -p "${WORK_DIR}"
    echo "Submitting a networked GPU job. Launcher: ${LAUNCHER}"
    # Slurm copies this file into its spool, so pass the real launcher path through.
    SOURCE_BUILD="${SOURCE_BUILD:-0}"
    RUN_LABELS="${RUN_LABELS:-}"
    export NIGHTLY_LAUNCHER="${LAUNCHER}" WORK_DIR TIER VARIANT SOURCE_BUILD RUN_LABELS
    exec sbatch --export=ALL \
        --output="${WORK_DIR}/slurm-%j.out" \
        --error="${WORK_DIR}/slurm-%j.err" \
        "${BASH_SOURCE[0]}"
fi

args=(
    --work-dir "${WORK_DIR}"
    --variant "${VARIANT}"
    --tier "${TIER}"
    --jobs "${SLURM_CPUS_PER_TASK:-32}"
)
if [[ "${SOURCE_BUILD:-0}" == "1" ]]; then
    args+=(--build-from-source)
fi
if [[ -n "${RUN_LABELS:-}" ]]; then
    args+=(--run-labels "${RUN_LABELS}")
fi

echo "host=$(hostname) job=${SLURM_JOB_ID} nodes=${SLURM_JOB_NODELIST:-unset}"
echo "gpus=${SLURM_JOB_GPUS:-unset} rocr=${ROCR_VISIBLE_DEVICES:-unset}"
python3 "${LAUNCHER}" "${args[@]}"
rc=$?
echo "result: $(cat "${WORK_DIR}/latest-result.txt" 2>/dev/null || echo missing)"
exit "${rc}"
