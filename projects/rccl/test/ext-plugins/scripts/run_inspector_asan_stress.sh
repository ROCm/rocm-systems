#!/usr/bin/env bash
# *************************************************************************
#  * Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#  *
#  * See LICENSE.txt for license information
#  ************************************************************************
#
# Optional ASAN regression runner for NCCL inspector plugin (NCCL issue #2000).
#
# Builds the native collInfo lifecycle unit test and the inspector plugin with
# ASAN, then runs the native tests and (when RCCL_TEST env is configured) the
# ext-inspector comm lifecycle stress pytest subset.
#
# Usage:
#   ./scripts/run_inspector_asan_stress.sh
#
# Optional environment:
#   RCCL_ROOT          Path to RCCL source tree (default: auto-detect from script)
#   RUN_FUNCTIONAL=1   Also run pytest stress tests (requires GPU + MPI + builds)
#   RCCL_INSTALL_DIR   Required when RUN_FUNCTIONAL=1
#   OMPI_INSTALL_DIR   Required when RUN_FUNCTIONAL=1
#   RCCL_TESTS_DIR     Required when RUN_FUNCTIONAL=1

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXT_PLUGINS_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
RCCL_ROOT="${RCCL_ROOT:-$(cd "${EXT_PLUGINS_DIR}/../.." && pwd)}"
INSPECTOR_TEST_DIR="${RCCL_ROOT}/plugins/profiler/inspector/test"
INSPECTOR_DIR="${RCCL_ROOT}/plugins/profiler/inspector"

echo "=== NCCL inspector ASAN regression (issue #2000) ==="
echo "RCCL_ROOT=${RCCL_ROOT}"

echo "--- Native collInfo lifecycle unit test (ASAN) ---"
make -C "${INSPECTOR_TEST_DIR}" clean
make -C "${INSPECTOR_TEST_DIR}" ASAN=1
make -C "${INSPECTOR_TEST_DIR}" ASAN=1 test
make -C "${INSPECTOR_TEST_DIR}" clean

echo "--- Inspector plugin ASAN build ---"
make -C "${INSPECTOR_DIR}" clean
# Inspector Makefile uses CUDA; skip plugin ASAN build if CUDA_HOME is unset.
if [[ -n "${CUDA_HOME:-}" ]] || [[ -d /usr/local/cuda ]]; then
  make -C "${INSPECTOR_DIR}" ASAN=1
else
  echo "Skipping inspector plugin ASAN build (CUDA_HOME not set)."
fi

if [[ "${RUN_FUNCTIONAL:-0}" == "1" ]]; then
  echo "--- Functional comm lifecycle stress (pytest, requires GPU cluster) ---"
  for var in RCCL_INSTALL_DIR OMPI_INSTALL_DIR RCCL_TESTS_DIR; do
    if [[ -z "${!var:-}" ]]; then
      echo "ERROR: ${var} must be set when RUN_FUNCTIONAL=1" >&2
      exit 1
    fi
  done
  cd "${EXT_PLUGINS_DIR}"
  if [[ ! -d venv ]]; then
    python3 -m venv venv
  fi
  # shellcheck disable=SC1091
  source venv/bin/activate
  pip install -q pytest
  pytest -m "ext_inspector and inspector_regression" tests/ext-inspector/test_lifecycle_stress.py -v --cache-clear
else
  echo "Functional stress skipped (set RUN_FUNCTIONAL=1 to enable)."
fi

echo "=== Inspector ASAN regression complete ==="
