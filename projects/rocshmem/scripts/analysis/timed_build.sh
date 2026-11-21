#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
###############################################################################
#
# Wraps any scripts/build_configs/* build script with wall-clock stage timing
# and a post-build ninja edge breakdown, without touching that script or
# _common.sh. Run from an empty build directory, same convention as the
# build-config scripts themselves:
#
#   mkdir build && cd build
#   ../scripts/analysis/timed_build.sh                         # defaults to all_backends
#   ../scripts/analysis/timed_build.sh ipc_single -DGPU_TARGETS=gfx950
#   ROCSHMEM_BUILD_TIMING=0 ../scripts/analysis/timed_build.sh  # disable timing, just run the build
#
# The first argument may be a build-config name (looked up under
# scripts/build_configs/, like all_backends does for itself) or a path to any
# executable build script with the same calling convention (cmake-args... in,
# built in cwd). Everything else is forwarded to it verbatim -- e.g.
# -DROCSHMEM_BUILD_TIME_TRACE=ON, -DGPU_TARGETS=..., which all_backends already
# passes straight through to cmake.
#
# Unlike the configure/build/install split a previous version of this
# tooling added directly to _common.sh, this measures only two wall-clock
# numbers without requiring any change to the wrapped script:
#   - total:        the whole wrapped invocation (configure + build + install)
#   - build (ninja): exact, read back out of the .ninja_log the build just
#                    produced
#   - configure+install (approx): total - build (ninja); not separated further
#     since this script has no hook into the wrapped script's internal cmake
#     calls to time them individually.
#
# Appends to build_timing.log in the cwd, then runs
# build_time_report.py --summary-only for the per-component/slowest-edges
# ninja breakdown.

set -e

script_dir=$(dirname "$(realpath "$0")")
build_configs_dir="${script_dir}/../build_configs"

if [[ "${ROCSHMEM_BUILD_TIMING:-1}" == 0 ]]; then
    exec "${build_configs_dir}/all_backends" "$@"
fi

build_script="${build_configs_dir}/all_backends"
if [[ $# -gt 0 && -x "${build_configs_dir}/$1" ]]; then
    build_script="${build_configs_dir}/$1"
    shift
elif [[ $# -gt 0 && -x "$1" ]]; then
    build_script="$1"
    shift
fi

timing_log=${ROCSHMEM_BUILD_TIMING_LOG:-build_timing.log}
: > "${timing_log}"

total_start=$(date +%s.%N)
"${build_script}" "$@"
total_end=$(date +%s.%N)
total_elapsed=$(awk -v s="${total_start}" -v e="${total_end}" 'BEGIN{printf "%.2f", e-s}')

build_elapsed="0.00"
if [[ -f .ninja_log ]]; then
    build_elapsed=$(python3 "${script_dir}/build_time_report.py" --build-dir . --summary-only 2>/dev/null \
        | awk -F'[: ]+' '/^ninja-tracked wall time/{print $4; exit}' | tr -d 's')
    build_elapsed=${build_elapsed:-0.00}
fi
other_elapsed=$(awk -v t="${total_elapsed}" -v b="${build_elapsed}" 'BEGIN{printf "%.2f", t-b}')

{
    printf '[timing] total                %ss\n' "${total_elapsed}"
    printf '[timing] build (ninja)        %ss\n' "${build_elapsed}"
    printf '[timing] configure+install (approx) %ss\n' "${other_elapsed}"
} | tee -a "${timing_log}"

if [[ -f .ninja_log ]] && command -v python3 >/dev/null 2>&1; then
    echo "[timing] ninja edge breakdown (build/${timing_log}):"
    python3 "${script_dir}/build_time_report.py" --build-dir . --summary-only \
        | tee -a "${timing_log}"
fi
