#!/usr/bin/env bash
# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
#
# Standalone check of the generated nccl.bindings struct wrappers, without
# uv, CMake or pytest: Cython-compile the four generated extension modules
# (nccl, cynccl, _internal.nccl, _internal.utils) from bindings/nccl4py into a
# scratch package and run tests/test_struct_buffer_semantics.py against it.
# The wrappers never call into librccl, so no GPU or RCCL build is needed.
#
# Usage: run_bindings_semantics.sh [test-name-regex]   (default: test_)
# Exit: 0 pass, 1 test failure, 2 build error, 77 prerequisites missing.
#
# Python with Cython>=3.1, numpy and setuptools is used as is; otherwise a
# venv is created under $WORK (uv if available, else python -m venv + pip).

set -u
PATTERN=${1:-test_}
HERE=$(cd "$(dirname "$0")" && pwd)
RCCL_DIR=${RCCL_DIR:-$(cd "$HERE/../.." && pwd)}
SRC=$RCCL_DIR/bindings/nccl4py
WORK=${NCCL4PY_STANDALONE_DIR:-${RCCL_BUILD:-$RCCL_DIR/build/release}/nccl4py_standalone}
PY=${PYTHON:-python3}

skip() { echo "SKIP: $*"; exit 77; }

command -v "$PY" >/dev/null || skip "$PY not found"
mkdir -p "$WORK" || skip "cannot create $WORK"
NEED='import Cython, numpy, setuptools; assert int(Cython.__version__.split(".")[0]) >= 3'
if ! "$PY" -c "$NEED" 2>/dev/null; then
  VENV=$WORK/venv
  if ! "$VENV/bin/python" -c "$NEED" 2>/dev/null; then
    rm -rf "$VENV"
    if command -v uv >/dev/null; then
      uv venv -q --python "$PY" "$VENV" &&
        uv pip install -q --python "$VENV/bin/python" 'Cython>=3.1' numpy setuptools
    else
      "$PY" -m venv "$VENV" &&
        "$VENV/bin/python" -m pip install -q 'Cython>=3.1' numpy setuptools
    fi || skip "cannot provide Cython>=3.1, numpy and setuptools"
  fi
  PY=$VENV/bin/python
fi
"$PY" -c 'import os, sys, sysconfig
sys.exit(not os.path.isfile(os.path.join(sysconfig.get_paths()["include"], "Python.h")))' ||
  skip "Python.h not found for $PY"
command -v "${CXX:-c++}" >/dev/null || command -v g++ >/dev/null || skip "no C++ compiler"

# Fresh scratch package per run: only the generated binding sources, with an
# empty nccl/__init__.py so the HIP cuda.core shim (hip-python) is not needed.
PKG=$(mktemp -d "$WORK/pkg.XXXXXX") || skip "mktemp failed"
trap 'rm -rf "$PKG"' EXIT
mkdir -p "$PKG/nccl/bindings/_internal"
cp "$SRC"/nccl/bindings/nccl.pyx "$SRC"/nccl/bindings/nccl.pxd \
   "$SRC"/nccl/bindings/cynccl.pyx "$SRC"/nccl/bindings/cynccl.pxd "$PKG/nccl/bindings/"
cp "$SRC"/nccl/bindings/_internal/*.pyx "$SRC"/nccl/bindings/_internal/*.pxd \
   "$SRC"/nccl/bindings/_internal/__init__.py "$PKG/nccl/bindings/_internal/"
: > "$PKG/nccl/__init__.py"
echo 'from .nccl import *' > "$PKG/nccl/bindings/__init__.py"
cat > "$PKG/setup.py" <<'EOF'
from Cython.Build import cythonize
from setuptools import Extension, setup

def ext(name, pyx):
    return Extension(name, [pyx], language="c++", extra_compile_args=["-std=c++14"], libraries=["dl"])

setup(name="nccl4py_standalone", ext_modules=cythonize([
    ext("nccl.bindings.nccl", "nccl/bindings/nccl.pyx"),
    ext("nccl.bindings.cynccl", "nccl/bindings/cynccl.pyx"),
    ext("nccl.bindings._internal.nccl", "nccl/bindings/_internal/nccl_linux.pyx"),
    ext("nccl.bindings._internal.utils", "nccl/bindings/_internal/utils.pyx"),
], language_level=3, nthreads=4))
EOF
if ! (cd "$PKG" && "$PY" setup.py -q build_ext --inplace -j 4 > build.log 2>&1); then
  tail -40 "$PKG/build.log"
  echo "ERROR: building nccl.bindings failed"
  exit 2
fi

cd "$PKG" && PYTHONPATH="$PKG" "$PY" "$SRC/tests/test_struct_buffer_semantics.py" "$PATTERN"
rc=$?
[ $rc -le 1 ] || exit 2
exit $rc
