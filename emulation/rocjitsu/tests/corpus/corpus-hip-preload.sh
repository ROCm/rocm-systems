#!/usr/bin/env bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Add HIP to the child environment prepared by the Clang ASan launcher, then
# replace this helper with the corpus command.

set -euo pipefail

if (( $# < 2 )); then
  echo "Usage: $0 HIP_RUNTIME COMMAND [ARGUMENT ...]" >&2
  exit 2
fi

hip_runtime="$1"
shift

if [[ "${hip_runtime}" != /* || ! -f "${hip_runtime}" || "${hip_runtime}" == *:* ]]; then
  echo "Invalid HIP runtime path: ${hip_runtime}" >&2
  exit 2
fi

preload="${LD_PRELOAD-}"
IFS=: read -r -a libraries <<< "${preload}"
if (( ${#libraries[@]} != 2 )); then
  echo "Expected Clang ASan and the rocjitsu interposer in LD_PRELOAD" >&2
  exit 2
fi
if [[ "${libraries[0]##*/}" != libclang_rt.asan*.so ||
      "${libraries[1]##*/}" != librocjitsu.so ]]; then
  echo "Unexpected Clang ASan corpus preload order: ${preload}" >&2
  exit 2
fi

export LD_PRELOAD="${preload}:${hip_runtime}"
exec -- "$@"
