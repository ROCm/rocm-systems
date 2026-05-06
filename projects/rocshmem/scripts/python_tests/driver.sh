#!/bin/bash
###############################################################################
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to
# deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or
# sell copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS
# IN THE SOFTWARE.
###############################################################################
#
# Driver script for rocshmem4py Python binding tests.
# Mirrors the functional_tests driver.sh pattern.
#
# Usage:
#   rocshmem_python_driver.sh <python_src_dir> <test_type> <log_dir> [hostfile]
#
#   python_src_dir : Path to rocshmem python/ source directory (for pip install)
#   test_type      : "all", "basic", "collective", "raw"
#                    - raw:        torch-free ctypes/HIP smoke tests
#                                  (runs on torch-less hosts)
#                    - basic:      single-PE tensor API tests  (require torch)
#                    - collective: multi-PE tensor API tests   (require torch)
#                    - all:        raw + basic + collective
#   log_dir        : Directory for test output logs
#   hostfile       : (optional) MPI hostfile for multi-node
#
# Example:
#   ./rocshmem_python_driver.sh /path/to/rocshmem/python all /tmp/python_logs
#
# NOTE: do NOT add `set -e` here. The functional_tests/ and unit_tests/
# drivers in this repo deliberately omit it so that per-test failures can be
# captured (`if [ $? -ne 0 ]; then cat $LOG; ...`) rather than aborting the
# whole script and hiding the log.

DRIVER_RETURN_STATUS=0
FAILED_LIST=""

ValidateInput() {
  if [ $1 -lt 3 ]; then
    echo "Usage: $0 <python_src_dir> <test_type> <log_dir> [hostfile]"
    echo "  python_src_dir : Path to rocshmem python/ source directory"
    echo "  test_type      : all, basic, collective, raw"
    echo "  log_dir        : Directory for test output logs"
    echo "  hostfile       : (optional) MPI hostfile"
    exit 1
  fi
}

ValidateLogDir() {
  if [ ! -d "$1" ]; then
    echo "LOG_DIR=$1 does not exist"
    mkdir -p "$1"
    echo "Created $1"
  fi
}

DetectGPUs() {
  if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null; then
    NUM_GPUS=${NUM_GPUS:-$(amd-smi list | grep GPU | wc -l)}
  elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null; then
    NUM_GPUS=${NUM_GPUS:-$(rocm-smi --showserial | grep GPU | wc -l)}
  fi
  NUM_GPUS=${NUM_GPUS:-0}
  NUM_GPUS=$(($NUM_GPUS > 0 ? $NUM_GPUS : 8))
}

ExecPythonTest() {
  local TEST_NAME=$1
  local NUM_RANKS=$2
  local TEST_FILES=$3

  local HEAP_SIZE=$((512 * 1024 * 1024))
  local TIMEOUT=$((5 * 60))

  if [ $NUM_GPUS -lt $NUM_RANKS ] && [ -z "$HOSTFILE" ]; then
    echo "Skip:   python_${TEST_NAME}_n${NUM_RANKS} ($NUM_RANKS > $NUM_GPUS GPUs)"
    return
  fi

  local -a cmd
  cmd=( mpirun
        --allow-run-as-root
        -n "$NUM_RANKS"
        ${OMPI_MCA_pml:+-mca pml "$OMPI_MCA_pml"}
        ${OMPI_MCA_osc:+-mca osc "$OMPI_MCA_osc"}
        -x "ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
        -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=${UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS:-16384}"
        -x "LD_LIBRARY_PATH"
        -x "WORLD_SIZE=$NUM_RANKS"
        -x "ROCSHMEM_USE_TORCH_INIT=0"
        ${TIMEOUT:+--timeout "$TIMEOUT"}
        ${HOSTFILE:+--hostfile "$HOSTFILE"}
        --map-by numa
        # DEBUG (drop before merge): -s disables pytest capture so prints/tracebacks
        # reach the console before prterun kills peers; --tb=long for full frames.
        pytest $TEST_FILES -v -s --tb=long -rA --color=no
      )

  local TEST_LOG_NAME="python_${TEST_NAME}_n${NUM_RANKS}"
  echo "Test:   $TEST_LOG_NAME"
  echo "# ${cmd[*]}" > "$LOG_DIR/$TEST_LOG_NAME.log"

  "${cmd[@]}" 2>&1 | tee -a "$LOG_DIR/$TEST_LOG_NAME.log"
  if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "FAILED: $TEST_LOG_NAME"
    DRIVER_RETURN_STATUS=1
    FAILED_LIST="$FAILED_LIST $TEST_LOG_NAME"
  fi
}

TestBasic() {
  ExecPythonTest "basic" 2 "$PYTHON_SRC_DIR/tests/test_basic.py"
}

TestCollective() {
  ExecPythonTest "collective" 2 "$PYTHON_SRC_DIR/tests/test_collective.py"
}

# test_smoke.py is the torch-free smoke suite; runs even on torch-less hosts
# (e.g. current Jenkins) so the native library is exercised in every CI pass.
TestRaw() {
  ExecPythonTest "raw" 2 "$PYTHON_SRC_DIR/tests/test_smoke.py"
}

TestAll() {
  ExecPythonTest "all" 2 "$PYTHON_SRC_DIR/tests/test_smoke.py $PYTHON_SRC_DIR/tests/test_basic.py $PYTHON_SRC_DIR/tests/test_collective.py"
}

# --- Main ---

ValidateInput $#

PYTHON_SRC_DIR=$1
TEST=$2
LOG_DIR=$3
HOSTFILE=$4

ValidateLogDir "$LOG_DIR"
DetectGPUs

# Ensure rocshmem4py is installed
if ! python3 -c "import rocshmem4py" 2>/dev/null; then
  echo "Installing rocshmem4py from $PYTHON_SRC_DIR ..."
  pip install -e "$PYTHON_SRC_DIR" || { echo "pip install failed"; exit 1; }
fi

echo "Python tests: type=$TEST, GPUs=$NUM_GPUS"
echo ""

case $TEST in
  "all")
    TestAll
    ;;
  "basic")
    TestBasic
    ;;
  "collective")
    TestCollective
    ;;
  "raw")
    TestRaw
    ;;
  *)
    echo "Unknown test type: $TEST"
    echo "Valid types: all, basic, collective, raw"
    exit 1
    ;;
esac

if [ "$DRIVER_RETURN_STATUS" -eq 0 ]; then
  echo "PYTHON TESTS PASSED"
else
  echo "PYTHON TESTS FAILED:$FAILED_LIST"
fi
exit $DRIVER_RETURN_STATUS
