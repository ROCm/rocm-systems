#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
set -euo pipefail

# Compare one real GFX1201 dispatch through this checkout and installed ROCr.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../../../.." && pwd)
test_dir=$(mktemp -d /tmp/rocddi-hsa-kernel.XXXXXX)
trap 'rm -rf -- "$test_dir"' EXIT

if [[ $# -ne 1 || ! -f "$1" ]]; then
  printf 'usage: %s ROCDDI_SHARED_LIBRARY\n' "$0" >&2
  exit 2
fi
library=$1
reference=${ROCR_REFERENCE_LIB:-/opt/rocm/lib/libhsa-runtime64.so.1}
bundler=${CLANG_OFFLOAD_BUNDLER:-${ROCM_PATH:-/opt/rocm}/llvm/bin/clang-offload-bundler}

hipcc --genco --offload-arch=gfx1201 -O2 \
  "$script_dir/kernel_dispatch.hip" -o "$test_dir/kernel.bundle"
"$bundler" -unbundle -type=o \
  -targets=hipv4-amdgcn-amd-amdhsa--gfx1201 \
  -input="$test_dir/kernel.bundle" -output="$test_dir/kernel.hsaco"
cc -std=gnu11 -O2 -Wall -Wextra -Werror \
  -I "$workspace_root/api-headers/include" \
  "$script_dir/kernel_dispatch_smoke.c" -ldl \
  -o "$test_dir/kernel_dispatch_smoke"

rocddi_result=$(env -u LD_PRELOAD timeout 40s \
  "$test_dir/kernel_dispatch_smoke" "$library" "$test_dir/kernel.hsaco")
rocr_result=$(env -u LD_PRELOAD timeout 40s \
  "$test_dir/kernel_dispatch_smoke" "$reference" "$test_dir/kernel.hsaco")
if [[ "$rocddi_result" != "$rocr_result" ]]; then
  printf 'HSA workload mismatch: rocddi=%s ROCr=%s\n' \
    "$rocddi_result" "$rocr_result" >&2
  exit 1
fi
printf 'rocddi and ROCr: %s\n' "$rocddi_result"
sha256sum "$library" "$reference" "$test_dir/kernel.hsaco" \
  "$test_dir/kernel_dispatch_smoke"
