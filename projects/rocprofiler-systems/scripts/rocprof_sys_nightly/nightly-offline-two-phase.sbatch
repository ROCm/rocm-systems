#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Two-phase nightly run for a cluster whose GPU nodes cannot reach the network.
#
# Phase 1 runs on the node where you launch this file (typically a login node):
# it downloads the tarball, clones the tests, and builds the venv.
# Phase 2 is the Slurm job below: it only reuses that staging area (--offline).
#
# Do not submit this file with sbatch yourself. Run it on the networked node:
#
#   VARIANT=gfx90a WORK_DIR=/lustre/$USER/rocprof-sys-nightly \
#     bash nightly-offline-two-phase.sbatch
#
# VARIANT is required. Login nodes usually have no GPU, so --variant auto cannot
# see the device. Use the tarball family for the compute-node GPU
# (gfx90a, gfx94X-dcgpu, ...).
# WORK_DIR must be on a filesystem both phases can read, and must not be
# world-writable. Edit the #SBATCH block for the site GPU queue before launching.

#SBATCH --job-name=rocprof-sys-offline
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --gres=gpu:4
#SBATCH --mem=128G
#SBATCH --time=04:00:00
#SBATCH --partition=gpu

set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="${NIGHTLY_LAUNCHER:-${HERE}/../run-nightly-tarball-tests.py}"
WORK_DIR="${WORK_DIR:-${SCRATCH:-${HOME}}/rocprof-sys-nightly}"
TIER="${TIER:-standard}"

if [[ -z "${SLURM_JOB_ID:-}" ]]; then
    if [[ -z "${VARIANT:-}" ]]; then
        echo "Set VARIANT to the compute-node GPU family, for example gfx90a or gfx94X-dcgpu." >&2
        exit 1
    fi
    if [[ ! -f "${LAUNCHER}" ]]; then
        echo "Cannot find ${LAUNCHER}. Set NIGHTLY_LAUNCHER to run-nightly-tarball-tests.py." >&2
        exit 1
    fi
    mkdir -p "${WORK_DIR}"
    echo "Phase 1 on $(hostname): stage tarball, source, and venv under ${WORK_DIR}"
    python3 "${LAUNCHER}" \
        --work-dir "${WORK_DIR}" \
        --variant "${VARIANT}" \
        --prepare-only

    echo "Phase 2: submitting the offline GPU job"
    # Slurm copies this file into its spool, so pass the real launcher path through.
    export NIGHTLY_LAUNCHER="${LAUNCHER}" WORK_DIR VARIANT TIER
    exec sbatch --export=ALL \
        --output="${WORK_DIR}/slurm-%j.out" \
        --error="${WORK_DIR}/slurm-%j.err" \
        "${BASH_SOURCE[0]}"
fi

echo "Phase 2 on $(hostname) job ${SLURM_JOB_ID}: offline --tier ${TIER}"
echo "nodes=${SLURM_JOB_NODELIST:-unset} gpus=${SLURM_JOB_GPUS:-unset}"
python3 "${LAUNCHER}" \
    --work-dir "${WORK_DIR}" \
    --offline \
    --tier "${TIER}"
rc=$?
echo "result: $(cat "${WORK_DIR}/latest-result.txt" 2>/dev/null || echo missing)"
exit "${rc}"
