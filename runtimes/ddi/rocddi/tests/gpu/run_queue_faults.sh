#!/usr/bin/env bash
# SPDX-License-Identifier: MIT
set -euo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
workspace_root=$(cd -- "$script_dir/../../../.." && pwd)
headers="$workspace_root/api-headers/include"
if [[ $# -ne 1 || ! -d "$1" ]]; then
  printf 'usage: %s LIBRARY_DIRECTORY\n' "$0" >&2
  exit 2
fi
library_dir=$(cd -- "$1" && pwd)
test_dir=$(mktemp -d /tmp/rocddi-queue-faults.XXXXXX)
trap 'rm -rf -- "$test_dir"' EXIT

cc -std=gnu11 -fPIC -shared -O2 -Wall -Wextra -Werror -pthread \
  -I "$headers" \
  "$script_dir/fault_ioctl.c" -o "$test_dir/fault_ioctl.so"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -I "$headers" \
  "$script_dir/fault_hsa.c" "$library_dir/libhsa_runtime64.so" \
  -ldl -Wl,-rpath,"$library_dir" -o "$test_dir/fault_hsa"
cc -std=gnu11 -O2 -Wall -Wextra -Werror -I "$headers" \
  "$script_dir/fault_amdf.c" -L "$library_dir" -lamdf \
  -ldl -Wl,-rpath,"$library_dir" -o "$test_dir/fault_amdf"

for pair in "fault_hsa libhsa-runtime64.so.1" "fault_amdf libhsa-runtime64.so.1"; do
  read -r probe soname <<< "$pair"
  resolved=$(env -u LD_PRELOAD LD_LIBRARY_PATH="$library_dir" \
    ldd "$test_dir/$probe" | awk -v name="$soname" '$1 == name { print $3 }')
  if [[ "$resolved" != "$library_dir/$soname" ]]; then
    printf 'wrong library for %s: %s\n' "$probe" "$resolved" >&2
    exit 1
  fi
done

for scenario in create mask; do
  timeout 30s env LD_PRELOAD="$test_dir/fault_ioctl.so" \
    LD_LIBRARY_PATH="$library_dir" "$test_dir/fault_hsa" "$scenario"
done
for scenario in create doorbell; do
  timeout 30s env LD_PRELOAD="$test_dir/fault_ioctl.so" \
    LD_LIBRARY_PATH="$library_dir" "$test_dir/fault_amdf" "$scenario"
done
