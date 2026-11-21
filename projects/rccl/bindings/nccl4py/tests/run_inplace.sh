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
# packaging and pytest: $PYTHON or the first python3* on PATH that has them,
# else a venv set up once in $NCCL4PY_TEST_DIR (default $RCCL_BUILD/nccl4py-test,
# else ~/.cache/nccl4py-test). The venv comes from pip and a python3 with
# headers, else from uv (found, pip-installed or downloaded), which can fetch a
# Python with headers. Exits 77 when none works and 2 when the build fails.
# hip-python is optional: these tests make no HIP calls.
set -uo pipefail

here=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
pkg=$(dirname "$here")
deps=(--only-binary=:all: "Cython>=3.1" numpy setuptools packaging pytest)  # wheels only: fail fast
case ${RCCL_BUILD:-} in "" | /*) build=${RCCL_BUILD:-} ;; *) build=${RCCL_SRC:-.}/$RCCL_BUILD ;; esac
dir=${NCCL4PY_TEST_DIR:-${build:+$build/nccl4py-test}}
dir=${dir:-${XDG_CACHE_HOME:-${HOME:-/tmp}/.cache}/nccl4py-test}
mkdir -p "$dir" && dir=$(cd "$dir" && pwd) || exit 2
venv=$dir/venv

headers() {  # python >= 3.10 with Python.h
  "$1" -c 'import os, sys, sysconfig
sys.exit(sys.version_info < (3, 10) or not os.path.exists(sysconfig.get_paths()["include"] + "/Python.h"))' \
    2> /dev/null
}
usable() {
  headers "$1" && "$1" -c 'import re, Cython, numpy, packaging, pytest, setuptools
assert tuple(int(x) for x in re.findall(r"\d+", Cython.__version__)[:2]) >= (3, 1)' 2> /dev/null
}

found=()
for c in ${PYTHON:-} python3 python3.14 python3.13 python3.12 python3.11 python3.10; do
  c=$(command -v "$c") || continue
  case $c in /*) found+=("$c") ;; *) found+=("$PWD/$c") ;; esac
done

setup() {  # creates $venv by the first route that works; logs to stdout
  local c uv="" bases=() seen=" "
  export PIP_DISABLE_PIP_VERSION_CHECK=1 PIP_CACHE_DIR=$dir/pip-cache UV_CACHE_DIR=$dir/uv-cache \
    UV_PYTHON_INSTALL_DIR=$dir/python
  echo "== $(date): found ${found[*]:-no python}"
  for c in "${found[@]}"; do
    case $seen in *" $(readlink -f "$c") "*) continue ;; esac
    seen+="$(readlink -f "$c") "
    headers "$c" && bases+=("$c")
  done
  for c in "${bases[@]}"; do
    echo "== venv + pip from $c"
    rm -rf "$venv"
    "$c" -m venv "$venv" && "$venv/bin/python" -m pip install -q --timeout 30 --retries 1 "${deps[@]}" &&
      return
  done
  for c in uv ~/.local/bin/uv ~/.cargo/bin/uv $(find "$dir/uv" -name uv -type f 2> /dev/null); do
    uv=$(command -v "$c") && break
  done
  for c in "${found[@]}"; do
    [ -z "$uv" ] || break
    echo "== pip install uv with $c"
    "$c" -m pip install -q --timeout 30 --retries 1 --target "$dir/uv" uv && uv=$dir/uv/bin/uv
  done
  if [ -z "$uv" ]; then
    c=https://github.com/astral-sh/uv/releases/latest/download/uv-$(uname -m)-unknown-linux-gnu.tar.gz
    echo "== download $c"
    mkdir -p "$dir/uv" && { curl -fsSL --connect-timeout 30 "$c" || wget -qO- "$c"; } |
      tar -xz -C "$dir/uv" && uv=$(find "$dir/uv" -name uv -type f | head -n 1)
  fi
  [ -n "$uv" ] || return 1
  # The last option set makes uv fetch a Python of its own, which has headers.
  for c in "${bases[@]/#/--python=}" "--python-preference=only-managed --python=3.12"; do
    echo "== $uv venv $c"
    rm -rf "$venv"
    # shellcheck disable=SC2086
    "$uv" venv -q $c "$venv" && "$uv" pip install -q --python "$venv/bin/python" "${deps[@]}" && return
  done
  return 1
}

py=
for c in "${found[@]}"; do usable "$c" && { py=$c; break; }; done
if [ -z "$py" ] && ! usable "$venv/bin/python"; then
  echo "setting up a test venv in $dir (log: $dir/setup.log)" >&2
  (command -v flock > /dev/null && flock 9; usable "$venv/bin/python" || setup) 9> "$dir/.lock" \
    >> "$dir/setup.log" 2>&1
fi
if [ -z "$py" ] && usable "$venv/bin/python"; then py=$venv/bin/python; fi
if [ -z "$py" ]; then
  echo "SKIP: no python with the prerequisites for building nccl4py in place" >&2
  tail -n 20 "$dir/setup.log" >&2 2> /dev/null
  exit 77
fi

# setuptools compiles with $CC/$CXX, else the interpreter's; fall back to an installed pair.
read -r cc cxx < <("$py" -c 'import sysconfig as s
print(*[(s.get_config_var(v) or "-").split()[0] for v in ("CC", "CXX")])')
cc=${CC:-$cc} cxx=${CXX:-$cxx}
if ! command -v "${cc%% *}" > /dev/null || ! command -v "${cxx%% *}" > /dev/null; then
  rocm=${ROCM_PATH:-/opt/rocm}/llvm/bin
  for c in gcc:g++ clang:clang++ "$rocm/clang:$rocm/clang++"; do
    if command -v "${c%%:*}" > /dev/null && command -v "${c#*:}" > /dev/null; then
      export CC=${c%%:*} CXX=${c#*:} LDSHARED="${c%%:*} -shared"
      break
    fi
  done
fi

work=$(mktemp -d "$dir/run.XXXXXX") || exit 2
trap 'rm -rf "$work"' EXIT
cp -r "$pkg/nccl" "$pkg/setup.py" "$work/"
find "$work/nccl/bindings" \( -name '*.so' -o -name '*.cpp' -o -name '*.c' \) -delete
# Trees whose nccl/__init__.py still imports the generated nccl/_version.py.
[ -e "$work/nccl/_version.py" ] || echo '__version__ = "0+inplace"' > "$work/nccl/_version.py"
if ! (cd "$work" && "$py" setup.py -q build_ext --inplace -j "$(nproc)") > "$work/build.log" 2>&1; then
  tail -n 40 "$work/build.log" >&2
  exit 2
fi

if ! "$py" -c 'from hip import hip' 2> /dev/null; then
  mkdir -p "$work/hipstub/hip"
  touch "$work/hipstub/hip/__init__.py" "$work/hipstub/hip/hip.py"
  export PYTHONPATH="$work/hipstub${PYTHONPATH:+:$PYTHONPATH}"
fi
export PYTHONPATH="$work${PYTHONPATH:+:$PYTHONPATH}" PYTHONDONTWRITEBYTECODE=1
cd "$here" && "$py" -m pytest -p no:cacheprovider "$@"
