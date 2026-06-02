#!/usr/bin/env bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
#
# mpiexec_deadlock_analysis.sh — Deploy attach_deadlock_analysis.sh to every node
#   in the MPI job via mpiexec (one rank per node), collect results into a
#   shared output directory, then perform a cross-rank coalescing pass that
#   highlights backtrace patterns uncommon among ranks.
#
# Usage:
#   ./mpiexec_deadlock_analysis.sh <executable_name> [options] [-- <mpiexec-args>...]
#
# Options:
#   --directory <dir>      Shared output directory visible from all nodes
#                          (default: ./rocshmem_deadlock_analysis_<timestamp>)
#   --cull                 Cull groups stuck in GPU barriers / gridsync
#   --cull=p1,p2,...       Cull groups matching any of the given substrings
#   --check-lanes          Enable per-lane register divergence check inside each group
#   --color                Force ANSI color output
#   --no-color             Disable ANSI color output
#   --no-coalesce          Skip the cross-rank coalescing pass
#   -- <mpiexec-args>...   All arguments after -- are forwarded verbatim to
#                          mpiexec (e.g. -np 4 -H node1,node2 --hostfile hosts.txt)
#
# mpiexec is invoked with -pernode, placing exactly one rank per allocated
# node.  The node set is determined by the scheduler allocation (SLURM, PBS)
# or by the arguments supplied after --.
#
# Environment variables:
#   MPIEXEC                mpiexec binary to use (default: mpiexec)
#   ROCGDB                 Path to rocgdb binary (default: rocgdb)
#   ROCSHMEM_GDB_TIMEOUT   Per-process timeout in seconds (default: 120)
#
# The output directory must reside on a shared filesystem (NFS, Lustre, etc.)
# accessible with the same path from all nodes.
#
# Examples:
#   ./mpiexec_deadlock_analysis.sh rocshmem_functional_tests \
#       --directory /shared/analysis --cull
#   # Explicit node list via mpiexec arguments:
#   ./mpiexec_deadlock_analysis.sh rocshmem_functional_tests \
#       --directory /shared/analysis -- -np 4 -H node1,node2,node3,node4
#   ./mpiexec_deadlock_analysis.sh rocshmem_functional_tests \
#       -- --hostfile /etc/mpi/hosts
###############################################################################

set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ATTACH_SCRIPT="${SCRIPT_DIR}/attach_deadlock_analysis.sh"
COALESCE_SCRIPT="${SCRIPT_DIR}/cross_rank_deadlock_analysis.py"

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

EXECUTABLE="${1:-}"
if [[ -z "${EXECUTABLE}" ]]; then
    echo "Usage: $0 <executable_name> [--directory <dir>] [--cull[=...]] [-- <mpiexec-args>...]" >&2
    exit 1
fi

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
OUTPUT_DIR=""
CULL_ARG=""
CHECK_LANES_ARG=""
COLOR_ARG=""
RUN_COALESCE=1
MPIEXEC_EXTRA_ARGS=()

_args=("${@:2}")
_i=0
while [[ ${_i} -lt ${#_args[@]} ]]; do
    _arg="${_args[${_i}]}"
    if [[ "${_arg}" == "--" ]]; then
        # Everything after -- is forwarded verbatim to mpiexec
        MPIEXEC_EXTRA_ARGS=("${_args[@]:$(( _i + 1 ))}")
        break
    elif [[ "${_arg}" == "--directory" ]]; then
        (( _i++ ))
        OUTPUT_DIR="${_args[${_i}]:-}"
        [[ -z "${OUTPUT_DIR}" ]] && { echo "ERROR: --directory requires an argument." >&2; exit 1; }
    elif [[ "${_arg}" == --directory=* ]]; then
        OUTPUT_DIR="${_arg#--directory=}"
        [[ -z "${OUTPUT_DIR}" ]] && { echo "ERROR: --directory= requires a value." >&2; exit 1; }
    elif [[ "${_arg}" == "--cull" ]]; then
        CULL_ARG="--cull"
    elif [[ "${_arg}" == --cull=* ]]; then
        CULL_ARG="${_arg}"
    elif [[ "${_arg}" == "--check-lanes" ]]; then
        CHECK_LANES_ARG="--check-lanes"
    elif [[ "${_arg}" == "--color" ]]; then
        COLOR_ARG="--color"
    elif [[ "${_arg}" == "--no-color" ]]; then
        COLOR_ARG="--no-color"
    elif [[ "${_arg}" == "--no-coalesce" ]]; then
        RUN_COALESCE=0
    else
        echo "ERROR: Unknown option '${_arg}'." >&2
        exit 1
    fi
    (( _i++ ))
done

OUTPUT_DIR="${OUTPUT_DIR:-./rocshmem_deadlock_analysis_${TIMESTAMP}}"
MPIEXEC="${MPIEXEC:-mpiexec}"

# ---------------------------------------------------------------------------
# Prerequisite checks
# ---------------------------------------------------------------------------

for _req in "${ATTACH_SCRIPT}" "${COALESCE_SCRIPT}"; do
    if [[ ! -f "${_req}" ]]; then
        echo "ERROR: Required script not found: ${_req}" >&2
        exit 1
    fi
done

if ! command -v "${MPIEXEC}" >/dev/null 2>&1; then
    echo "ERROR: '${MPIEXEC}' not found in PATH." >&2
    echo "       Set the MPIEXEC environment variable to the full path." >&2
    exit 1
fi

if ! command -v python3 >/dev/null 2>&1; then
    echo "ERROR: python3 not found in PATH (required for cross-rank coalescing)." >&2
    exit 1
fi

mkdir -p "${OUTPUT_DIR}" || { echo "ERROR: Cannot create output directory: ${OUTPUT_DIR}" >&2; exit 1; }

# ---------------------------------------------------------------------------
# Deploy attach_deadlock_analysis.sh on each node via mpiexec
# ---------------------------------------------------------------------------

echo "Deploying deadlock analysis..."
echo "Executable:       ${EXECUTABLE}"
echo "Output directory: ${OUTPUT_DIR}"
[[ -n "${CULL_ARG}" ]]         && echo "Cull option:      ${CULL_ARG}"
[[ -n "${CHECK_LANES_ARG}" ]]  && echo "Check lanes:      yes"
[[ -n "${COLOR_ARG}" ]]        && echo "Color:            ${COLOR_ARG}"
[[ ${#MPIEXEC_EXTRA_ARGS[@]} -gt 0 ]] && echo "mpiexec args:     ${MPIEXEC_EXTRA_ARGS[*]}"
echo ""

# Build the attach script argument list
_attach_args=("${EXECUTABLE}" "--directory" "${OUTPUT_DIR}")
[[ -n "${CULL_ARG}" ]]        && _attach_args+=("${CULL_ARG}")
[[ -n "${CHECK_LANES_ARG}" ]] && _attach_args+=("${CHECK_LANES_ARG}")
[[ -n "${COLOR_ARG}" ]]       && _attach_args+=("${COLOR_ARG}")

# -pernode launches exactly one process per allocated node (supported by
# OpenMPI, MPICH, and Intel MPI).  ROCGDB and ROCSHMEM_GDB_TIMEOUT are
# forwarded automatically via the inherited environment.
# MPIEXEC_EXTRA_ARGS (from arguments after --) supply the node list / count,
# e.g. -np 4 -H node1,node2 or --hostfile hosts.txt.
"${MPIEXEC}" "${MPIEXEC_EXTRA_ARGS[@]}" -pernode \
    bash "${ATTACH_SCRIPT}" "${_attach_args[@]}"
MPIEXEC_EXIT=$?

echo ""
if [[ ${MPIEXEC_EXIT} -ne 0 ]]; then
    echo "WARNING: mpiexec exited with status ${MPIEXEC_EXIT}." \
         "Some nodes may have failed." >&2
fi

# ---------------------------------------------------------------------------
# Cross-rank coalescing pass
# ---------------------------------------------------------------------------

if [[ ${RUN_COALESCE} -eq 1 ]]; then
    echo "Running cross-rank coalescing pass..."
    echo ""
    _coalesce_args=("${OUTPUT_DIR}")
    [[ -n "${COLOR_ARG}" ]] && _coalesce_args+=("${COLOR_ARG}")
    python3 "${COALESCE_SCRIPT}" "${_coalesce_args[@]}"
fi
