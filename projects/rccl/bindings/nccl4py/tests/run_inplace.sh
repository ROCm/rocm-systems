#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Runs unit-style nccl4py tests without installing the wheel: builds the Cython
# extensions into a scratch copy of the package, then runs pytest from this
# directory against that copy. Arguments are passed to pytest:
#
#   bash run_inplace.sh -q test_comm_properties_facade.py
#
# Needs python >= 3.10 with headers, Cython >= 3.1, setuptools, numpy,
# packaging and pytest ($PYTHON, else python3, else an ephemeral uv
# environment). Exits 77 when none qualifies and 2 when the build fails.
# hip-python is optional: these tests make no HIP calls.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pkg=$(dirname "$here")
py=("${PYTHON:-python3}")

usable() {
  "${py[@]}" - 2>/dev/null <<'EOF'
import os, re, sys, sysconfig
import Cython, numpy, packaging, pytest, setuptools  # noqa: F401
assert sys.version_info >= (3, 10)
assert tuple(int(x) for x in re.findall(r"\d+", Cython.__version__)[:2]) >= (3, 1)
assert os.path.exists(os.path.join(sysconfig.get_paths()["include"], "Python.h"))
EOF
}
if ! usable && [ -z "${PYTHON:-}" ] && command -v uv > /dev/null; then
  py=(uv run -q --no-project --with "Cython>=3.1" --with numpy --with setuptools
      --with packaging --with pytest python)
fi
if ! usable; then
  echo "SKIP: no python with the prerequisites for building nccl4py in place" >&2
  exit 77
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
cp -r "$pkg/nccl" "$pkg/setup.py" "$work/"
find "$work/nccl/bindings" \( -name '*.so' -o -name '*.cpp' -o -name '*.c' \) -delete
# Trees whose nccl/__init__.py still imports the generated nccl/_version.py.
[ -e "$work/nccl/_version.py" ] || echo '__version__ = "0+inplace"' > "$work/nccl/_version.py"
if ! (cd "$work" && "${py[@]}" setup.py -q build_ext --inplace -j "$(nproc)") \
    > "$work/build.log" 2>&1; then
  tail -n 40 "$work/build.log" >&2
  exit 2
fi

if ! "${py[@]}" -c 'from hip import hip' 2> /dev/null; then
  mkdir -p "$work/hipstub/hip"
  touch "$work/hipstub/hip/__init__.py" "$work/hipstub/hip/hip.py"
  export PYTHONPATH="$work/hipstub${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$work${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
cd "$here" && "${py[@]}" -m pytest -p no:cacheprovider "$@"
