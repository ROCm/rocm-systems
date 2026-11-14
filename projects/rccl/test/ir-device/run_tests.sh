#!/usr/bin/env bash
# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
#
# Runs this pytest suite against an existing RCCL build, e.g. from CI:
#
#   RCCL_BUILD=/path/to/build/release bash run_tests.sh -k reduce_copy_typed
#
# Keeps ARCH when it is already a gfx target, else takes it from the first
# GPU, finds GoogleTest when the RCCL build did not download its own, and gets
# pytest from uv or a local .venv when python3 has none. Arguments go to
# pytest. Exits 0 when a test passed and none failed, 1 on a failed test, 2
# when a fixture (bitcode or IR_test build) errored and 77 (skipped) when
# nothing ran.

cd "$(dirname "${BASH_SOURCE[0]}")" || exit 2
unset RCCL_DIR  # derived from this checkout by conftest.py
if [ -z "${ROCM_PATH:-}" ]; then
  ROCM_PATH=/opt/rocm
  if [ ! -x "$ROCM_PATH/bin/hipcc" ] && command -v hipcc > /dev/null; then
    ROCM_PATH=$(dirname "$(dirname "$(readlink -f "$(command -v hipcc)")")")
  fi
fi
export ROCM_PATH

if [[ "${ARCH:-}" != gfx* ]]; then
  arch=$({ "$ROCM_PATH/bin/rocm_agent_enumerator" || "$ROCM_PATH/lib/llvm/bin/amdgpu-arch" ||
           "$ROCM_PATH/llvm/bin/amdgpu-arch"; } 2> /dev/null |
         grep -m1 '^gfx[1-9]')
  if [ -n "$arch" ]; then export ARCH="$arch"; else unset ARCH; fi
fi

build=${RCCL_BUILD:-../../build/release}
# Same test as conftest.py: the header plus libgtest under lib, lib64 or lib/<triple>.
has_gtest() {
  [ -f "$1/include/gtest/gtest.h" ] || return 1
  local lib
  for lib in "$1"/lib/libgtest.{a,so} "$1"/lib64/libgtest.{a,so} "$1"/lib/*-linux-gnu/libgtest.{a,so}; do
    [ -f "$lib" ] && return 0
  done
  return 1
}
if [ -z "${GTEST_ROOT:-}" ] && ! has_gtest "$build/gtest"; then
  for p in $(sed -n 's#^GTest_DIR:PATH=\(.*\)/lib[^/]*/.*$#\1#p' "$build/CMakeCache.txt" 2> /dev/null) \
           /usr /usr/local "$ROCM_PATH"; do
    if has_gtest "$p"; then export GTEST_ROOT="$p"; break; fi
  done
fi

py=(python3)
if ! python3 -c 'import pytest' 2> /dev/null; then
  if uv run -q --no-project --with pytest python3 -c 'import pytest' > /dev/null 2>&1; then
    py=(uv run -q --no-project --with pytest python3)
  elif { [ -x .venv/bin/python ] || python3 -m venv .venv; } > /dev/null 2>&1 &&
       { .venv/bin/python -c 'import pytest' || .venv/bin/python -m pip install -q pytest; } > /dev/null 2>&1; then
    py=(.venv/bin/python)
  elif PYTHONPATH=.venv/site python3 -c 'import pytest' 2> /dev/null ||
       python3 -m pip install -q --target .venv/site pytest > /dev/null 2>&1; then
    # No python3-venv: pip --target also skips the PEP 668 externally-managed check.
    export PYTHONPATH="$PWD/.venv/site${PYTHONPATH:+:$PYTHONPATH}"
  fi
  if ! "${py[@]}" -c 'import pytest' 2> /dev/null; then
    echo "SKIP: pytest is not available" >&2
    exit 77
  fi
fi

rc=0
out=$("${py[@]}" -m pytest -p no:cacheprovider -q -ra tests/test_ir_device.py "$@" 2>&1) || rc=$?
[ "$rc" -ne 0 ] && tail -n 20 logs/*.log 2> /dev/null
printf '%s\n' "$out"
summary=$(printf '%s\n' "$out" | grep -E '^=*[ ]*[0-9]+ (passed|failed|skipped|errors?|deselected)' | tail -n 1)
if [[ "$summary" =~ [0-9]+\ errors? ]]; then exit 2; fi
if [ "$rc" -eq 5 ]; then exit 77; fi  # pytest: no tests collected
if [ "$rc" -ne 0 ]; then exit "$rc"; fi
[[ "$summary" =~ [0-9]+\ passed ]] || exit 77
