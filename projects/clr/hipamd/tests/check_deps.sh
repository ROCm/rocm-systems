#!/bin/bash
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
#
# SPDX-License-Identifier: MIT
#
# Dependency guard for libamdhip64 (Linux, shared build).
#
#   check_deps.sh <libamdhip64.so> [needed_allowlist.txt]
#
# Fails when
#   * a DT_NEEDED entry is not matched by a regular expression of the allowlist (one per line,
#     '#' starts a comment), which also rejects libhsa-runtime64 and libamd_comgr: ROCr and comgr
#     are loaded with dlopen so that a process that selects another runtime never maps them;
#   * the library has an undefined hsa_* or amd_comgr_* symbol (a call that bypasses amd::roc::Hsa
#     or amd::Comgr, which would fail when the library is loaded without the link dependency).
set -u

lib=${1:?usage: check_deps.sh <libamdhip64.so> [needed_allowlist.txt]}
allow=${2:-$(dirname "$0")/needed_allowlist.txt}
fail=0

needed=$(readelf -d "$lib" | sed -n 's/.*(NEEDED).*Shared library: \[\(.*\)\]/\1/p')
patterns=$(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$allow")

for n in $needed; do
  ok=0
  while read -r re; do
    if [[ $n =~ $re ]]; then ok=1; break; fi
  done <<<"$patterns"
  if [ $ok -eq 0 ]; then
    echo "FAIL: unexpected DT_NEEDED entry: $n"
    fail=1
  fi
done

undef=$(nm -D --undefined-only "$lib" | awk '$2 ~ /^(hsa_|amd_comgr_)/ {print $2}')
if [ -n "$undef" ]; then
  echo "FAIL: undefined ROCr/comgr symbols (route them through amd::roc::Hsa / amd::Comgr):"
  echo "$undef" | sed 's/@.*//; s/^/  /' | head -10
  echo "  ($(echo "$undef" | wc -l) in total)"
  fail=1
fi

[ $fail -eq 0 ] && echo "OK: $lib: DT_NEEDED and undefined symbols match the allowlist"
exit $fail
