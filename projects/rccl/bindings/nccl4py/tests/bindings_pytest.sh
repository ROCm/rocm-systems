#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Builds this tree's nccl/bindings alone (under an empty nccl/__init__.py, so no
# hip-python) into a cache under $RCCL_BUILD and runs the given tests/ node ids
# against $RCCL_BUILD/librccl.so (else $NCCL_LIBRARY), e.g.
#   bash tests/bindings_pytest.sh test_bindings_version.py::test_control_query_fills_rank
# Exits with pytest's status, 77 when no Python with Cython>=3.1, numpy, pytest,
# setuptools and Python.h can be set up, and 2 when librccl.so is missing or the
# build fails.
set -u
[ $# -gt 0 ] || { echo "usage: $0 <test file>[::<test>]..."; exit 2; }
tests=$(cd "$(dirname "$0")" && pwd)
pkg=$(dirname "$tests")
case ${RCCL_BUILD:-} in "" | /*) ;; *) RCCL_BUILD=${RCCL_SRC:-.}/$RCCL_BUILD ;; esac
lib=${RCCL_BUILD:+$RCCL_BUILD/librccl.so}
lib=$(readlink -f "${lib:-${NCCL_LIBRARY:-}}")
[ -f "$lib" ] || { echo "ERROR: no librccl.so; set RCCL_BUILD or NCCL_LIBRARY"; exit 2; }
cache=${RCCL_BUILD:-${TMPDIR:-/tmp}}/nccl4py-bindings-pytest
mkdir -p "$cache" && cache=$(cd "$cache" && pwd) || exit 2
check='import os, sys, sysconfig, Cython, numpy, pytest, setuptools
ok = tuple(int(v) for v in Cython.__version__.split(".")[:2]) >= (3, 1)
sys.exit(not ok or not os.path.exists(sysconfig.get_paths()["include"] + "/Python.h"))'
py=${PYTHON:-python3}
if ! "$py" -c "$check" 2>/dev/null; then
  py=$cache/venv/bin/python
  if ! "$py" -c "$check" 2>/dev/null; then  # set up a private venv once
    rm -rf "$cache/venv"
    deps=("cython>=3.1" numpy pytest setuptools)
    if command -v uv >/dev/null; then
      uv venv -q "$cache/venv" && uv pip install -q --python "$py" "${deps[@]}"
    else
      "${PYTHON:-python3}" -m venv "$cache/venv" && "$py" -m pip install -q "${deps[@]}"
    fi >/dev/null 2>&1
    "$py" -c "$check" 2>/dev/null ||
      { echo "SKIP: no Python with Cython>=3.1, numpy, pytest, setuptools and Python.h"; exit 77; }
  fi
fi
srcs=$(cd "$pkg" && find setup.py nccl/bindings -name '*.py' -o -name '*.pyx' -o -name '*.pxd' |
  sort)
key=$( (cd "$pkg" && cat $srcs && "$py" -VV &&
  "$py" -c 'import Cython; print(Cython.__version__)') | sha1sum | cut -c1-16)
out=$cache/$key
if [ ! -e "$out/.built" ]; then
  stage=$(mktemp -d "$cache/stage.XXXXXX") || exit 2
  (cd "$pkg" && cp --parents $srcs "$stage") && touch "$stage/nccl/__init__.py" || exit 2
  if ! (cd "$stage" && CFLAGS="-O0 -g0" CXXFLAGS="-O0 -g0" \
        "$py" setup.py -q build_ext --inplace -j 8) >"$stage/build.log" 2>&1; then
    tail -n 40 "$stage/build.log"; rm -rf "$stage"
    echo "ERROR: building nccl.bindings failed"; exit 2
  fi
  touch "$stage/.built"
  mv -T "$stage" "$out" 2>/dev/null || rm -rf "$stage"  # a concurrent run built it first
fi
args=(); for a in "$@"; do args+=("$tests/$a"); done
export NCCL_LIBRARY="$lib" PYTHONPATH="$out" PYTHONDONTWRITEBYTECODE=1
LD_LIBRARY_PATH="$(dirname "$lib"):${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export LD_LIBRARY_PATH
cd "$out" && exec "$py" -m pytest -p no:cacheprovider -q "${args[@]}"
