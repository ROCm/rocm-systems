#!/usr/bin/env bash
#
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# Run a command only if SHMEM validation passed (marker file exists).
# Usage: run_if_shmem_ok.sh <marker_file> <command> [args...]
# Exit 77 if marker is missing (CTest SKIP_RETURN_CODE 77).

MARKER="$1"
shift
test -f "$MARKER" || exit 77
exec "$@"
