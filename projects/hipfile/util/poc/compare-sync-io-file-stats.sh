#!/usr/bin/env bash

# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") AIS_DIRECTORY [GPU_ID]

Build and run the synchronous hipFile/POSIX metadata comparison on an
AIS-capable filesystem. The matrix is run once with hipFile fastpath only and
once with hipFile fallback only.

Environment:
  HIPFILE_BUILD_DIR  hipFile build directory (default: <source>/build)
  ROCM_PATH          ROCm prefix (default: hipconfig --path, then /opt/rocm)
  CXX                C++ compiler (default: amdclang++)
  KEEP_POC_FILES=1   retain generated test files after a successful run
EOF
}

if [[ $# -lt 1 || $# -gt 2 ]]; then
    usage >&2
    exit 2
fi

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
project_dir=$(cd -- "$script_dir/../.." && pwd)
ais_directory=$(realpath -e -- "$1")
gpu_id=${2:-0}
build_dir=${HIPFILE_BUILD_DIR:-"$project_dir/build"}
compiler=${CXX:-amdclang++}
if [[ -n ${ROCM_PATH:-} ]]; then
    rocm_path=$ROCM_PATH
elif command -v hipconfig >/dev/null 2>&1; then
    rocm_path=$(hipconfig --path)
else
    rocm_path=/opt/rocm
fi

if [[ ! -d "$ais_directory" || ! -r "$ais_directory" || ! -w "$ais_directory" || ! -x "$ais_directory" ]]; then
    echo "AIS_DIRECTORY must be a readable, writable, searchable directory: $ais_directory" >&2
    exit 2
fi
if [[ ! "$gpu_id" =~ ^[0-9]+$ ]]; then
    echo "GPU_ID must be a non-negative integer: $gpu_id" >&2
    exit 2
fi
if ! command -v "$compiler" >/dev/null 2>&1; then
    echo "C++ compiler not found: $compiler" >&2
    exit 2
fi

cmake \
    -S "$project_dir" \
    -B "$build_dir" \
    -DCMAKE_CXX_COMPILER="$compiler" \
    -DBUILD_TESTING=OFF \
    -DAIS_INSTALL_EXAMPLES=OFF \
    -DAIS_INSTALL_TOOLS=ON
cmake --build "$build_dir" --parallel --target hipfile ais-stats

lib_dir="$build_dir/src/amd_detail"
binary_dir="$build_dir/util/poc"
binary="$binary_dir/compare-sync-io-file-stats"
mkdir -p -- "$binary_dir"

"$compiler" \
    -std=c++20 \
    -D__HIP_PLATFORM_AMD__ \
    -Wall \
    -Wextra \
    -Wpedantic \
    -I"$project_dir/include" \
    -isystem "$rocm_path/include" \
    "$script_dir/compare-sync-io-file-stats.cpp" \
    -L"$lib_dir" \
    -L"$rocm_path/lib" \
    -L"$rocm_path/lib64" \
    -Wl,-rpath,"$lib_dir" \
    -Wl,-rpath,"$rocm_path/lib" \
    -lhipfile \
    -lamdhip64 \
    -o "$binary"

run_root=$(mktemp -d "$ais_directory/hipfile-io-stat-effects.XXXXXX")
# shellcheck disable=SC2317 # Invoked by the EXIT trap.
cleanup() {
    local status=$?
    if [[ $status -ne 0 || ${KEEP_POC_FILES:-0} == 1 ]]; then
        echo "POC files retained at: $run_root" >&2
    else
        rm -rf -- "$run_root"
    fi
}
trap cleanup EXIT

echo "HOST"
hostname
id
echo
echo "FILESYSTEM"
findmnt -T "$run_root" -o TARGET,SOURCE,FSTYPE,OPTIONS
echo

if [[ -x "$project_dir/tools/ais-check/ais-check" ]]; then
    echo "AIS CHECK"
    "$project_dir/tools/ais-check/ais-check"
    echo
fi

stats_binary="$build_dir/tools/ais-stats/ais-stats"

run_backend() {
    local label=$1
    local allow_compat=$2
    local force_compat=$3
    local run_directory="$run_root/$label"
    local program_status
    local runtime_library_path
    local stats_status
    local target_pid
    mkdir -- "$run_directory"
    runtime_library_path="$lib_dir:$rocm_path/lib:$rocm_path/lib64:${LD_LIBRARY_PATH:-}"

    echo "#################### HIPFILE BACKEND: ${label^^} ####################"
    set +e
    env \
        LD_LIBRARY_PATH="$runtime_library_path" \
        HIPFILE_STATS_LEVEL=1 \
        HIPFILE_ALLOW_COMPAT_MODE="$allow_compat" \
        HIPFILE_FORCE_COMPAT_MODE="$force_compat" \
        "$binary" "$run_directory" "$gpu_id" &
    target_pid=$!
    env LD_LIBRARY_PATH="$runtime_library_path" "$stats_binary" -p "$target_pid"
    stats_status=$?
    wait "$target_pid"
    program_status=$?
    set -e

    if [[ $program_status -ne 0 ]]; then
        echo "$label POC process failed with status $program_status" >&2
        return "$program_status"
    fi
    if [[ $stats_status -ne 0 ]]; then
        echo "$label ais-stats failed with status $stats_status" >&2
        return "$stats_status"
    fi
    echo
}

overall_status=0
if ! run_backend fastpath false false; then
    overall_status=1
fi
if ! run_backend fallback true true; then
    overall_status=1
fi
exit "$overall_status"
