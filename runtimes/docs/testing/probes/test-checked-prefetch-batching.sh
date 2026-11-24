#!/bin/sh
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
# Exercise the actual patched kernel helpers with host-side fence mocks.
set -eu
if [ "$#" -ne 1 ]; then
    echo "usage: $0 PATCHED_AMDGPU_SOURCE" >&2
    exit 2
fi
probe_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
batch_tmp=$(mktemp -d)
trap 'rm -rf -- "$batch_tmp"' EXIT HUP INT TERM
sed -n '/^int amdgpu_ttm_zero_vram_checked(/,/^}/p' \
    "$1/amd/amdgpu/amdgpu_ttm.c" > "$batch_tmp/checked-prefetch-batching.inc"
sed -n '/^static int svm_migrate_zero_to_vram(/,/^}/p' \
    "$1/amd/amdkfd/kfd_migrate.c" >> "$batch_tmp/checked-prefetch-batching.inc"
"${CC:-cc}" -std=c11 -Wall -Wextra -Werror -I "$batch_tmp" \
    "$probe_dir/checked-prefetch-batching.c" -o "$batch_tmp/test"
"$batch_tmp/test"
