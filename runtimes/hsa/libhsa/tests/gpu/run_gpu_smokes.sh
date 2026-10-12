#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

# Opt-in GFX1201 hardware checks. Probe artifacts stay outside the checkout.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../../../.." && pwd)
if [[ $# -ne 1 || ! -d "$1" ]]; then
  printf 'usage: %s LIBRARY_DIRECTORY\n' "$0" >&2
  exit 2
fi
library_dir=$(cd -- "$1" && pwd)
test_dir=$(mktemp -d /tmp/rocddi-pcs-gpu.XXXXXX)
trap 'rm -rf -- "$test_dir"' EXIT

cc -std=gnu11 -O2 -Wall -Wextra -Werror \
  -I "$workspace_root/api-headers/include" \
  "$script_dir/pc_sampling_smoke.c" \
  "$library_dir/libhsa_runtime64.so" \
  -Wl,-rpath,"$library_dir" \
  -o "$test_dir/pc_sampling_smoke"

cc -std=gnu11 -O2 -Wall -Wextra -Werror \
  -I "$workspace_root/api-headers/include" \
  "$script_dir/callback_shutdown_smoke.c" \
  "$library_dir/libhsa_runtime64.so" \
  -Wl,-rpath,"$library_dir" \
  -o "$test_dir/callback_shutdown_smoke"

cc -std=gnu11 -O2 -Wall -Wextra -Werror \
  -I "$workspace_root/api-headers/include" \
  "$script_dir/file_logging_smoke.c" \
  "$library_dir/libhsa_runtime64.so" \
  -Wl,-rpath,"$library_dir" \
  -o "$test_dir/file_logging_smoke"

for probe in pc_sampling_smoke callback_shutdown_smoke file_logging_smoke; do
  resolved=$(env -u LD_PRELOAD LD_LIBRARY_PATH="$library_dir" \
    ldd "$test_dir/$probe" | awk '$1 == "libhsa-runtime64.so.1" { print $3 }')
  if [[ "$resolved" != "$library_dir/libhsa-runtime64.so.1" ]]; then
    printf 'wrong HSA library for %s: %s\n' "$probe" "$resolved" >&2
    exit 1
  fi
done

env -u LD_PRELOAD LD_LIBRARY_PATH="$library_dir" \
  timeout 30s "$test_dir/callback_shutdown_smoke"
env -u LD_PRELOAD LD_LIBRARY_PATH="$library_dir" \
  timeout 30s "$test_dir/pc_sampling_smoke"
logging_output=$(env -u LD_PRELOAD LD_LIBRARY_PATH="$library_dir" \
  timeout 30s "$test_dir/file_logging_smoke" 2>&1)
printf '%s\n' "$logging_output"
if [[ "$logging_output" != *"created AQL queue"* ]]; then
  printf 'stderr logging did not emit the queue creation record\n' >&2
  exit 1
fi
