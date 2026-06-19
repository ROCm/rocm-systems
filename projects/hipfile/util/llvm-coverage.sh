#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

set -euo pipefail

SCRIPT_PATH="$(realpath "$0")"
SCRIPT_DIR="$(dirname "$SCRIPT_PATH")"
BUILD_DIR="${SCRIPT_DIR}/../build"
COLOR_ARG=""

function usage() {
    echo -e "Usage:
  $0 [-b directory] [-c] [-h]

OPTIONS
  -b Path to build directory (defaults to <repo_root>/build)
  -c Show colors in output
  -h Show usage information"
}

while getopts "b:ch" o; do
    case "${o}" in
    b)
        BUILD_DIR="$OPTARG"
        ;;
    c)
        COLOR_ARG="-use-color"
        ;;
    h)
        usage
        exit
        ;;
    *)
        usage
        exit
        ;;
    esac
done

COV_FILES="$BUILD_DIR/coverage-files.txt"

find "$BUILD_DIR" -name "*.profraw" >"$COV_FILES"
if [ ! -s "$COV_FILES" ]; then
    echo "No coverage files found" >>/dev/stderr
    exit 1
fi

llvm-profdata merge -output="$BUILD_DIR"/coverage.profdata --input-files="$COV_FILES"
llvm-cov report "$COLOR_ARG" -instr-profile="$BUILD_DIR"/coverage.profdata -object="$BUILD_DIR"/src/amd_detail/libhipfile.so >"$BUILD_DIR"/coverage-report.txt
llvm-cov show "$COLOR_ARG" -instr-profile="$BUILD_DIR"/coverage.profdata -object="$BUILD_DIR"/src/amd_detail/libhipfile.so >"$BUILD_DIR"/coverage-lines.txt
