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
#   test_type      : "all", "basic", "collective"
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
    echo "  test_type      : all, basic, collective"
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
        -mca pml "${OMPI_MCA_pml:-ucx}"
        -mca osc "${OMPI_MCA_osc:-ucx}"
        -x "ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
        -x "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=${UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS:-16384}"
        -x "LD_LIBRARY_PATH"
        -x "WORLD_SIZE=$NUM_RANKS"
        -x "ROCSHMEM_USE_TORCH_INIT=0"
        ${TIMEOUT:+--timeout "$TIMEOUT"}
        ${HOSTFILE:+--hostfile "$HOSTFILE"}
        --map-by numa
        pytest $TEST_FILES -v
      )

  local TEST_LOG_NAME="python_${TEST_NAME}_n${NUM_RANKS}"
  echo "Test:   $TEST_LOG_NAME"
  echo "# ${cmd[*]}" > "$LOG_DIR/$TEST_LOG_NAME.log"

  # Use `if` so `set -e` does not terminate the script on mpirun failure,
  # otherwise the log is never cat'd and the real error stays hidden in CI.
  if ! "${cmd[@]}" >> "$LOG_DIR/$TEST_LOG_NAME.log" 2>&1; then
    echo "FAILED: $TEST_LOG_NAME"
    cat "$LOG_DIR/$TEST_LOG_NAME.log"
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

TestAll() {
  ExecPythonTest "all" 2 "$PYTHON_SRC_DIR/tests/test_basic.py $PYTHON_SRC_DIR/tests/test_collective.py"
}

# --- Main ---

ValidateInput $#

PYTHON_SRC_DIR=$1
TEST=$2
LOG_DIR=$3
HOSTFILE=$4

ValidateLogDir "$LOG_DIR"
DetectGPUs

# ---------------------------------------------------------------------------
# TEMP: remove before merge
# Print the active MPI/UCX environment BEFORE pip install, so any linkage
# mismatch is visible in the Jenkins log even when the test itself fails.
# ---------------------------------------------------------------------------
echo "=== TEMP MPI/UCX env diagnostics (remove before merge) ==="
{
  echo "--- which mpirun / version ---"
  command -v mpirun && mpirun --version 2>&1 | head -3
  echo "--- PATH ---"; echo "$PATH"
  echo "--- LD_LIBRARY_PATH ---"; echo "${LD_LIBRARY_PATH:-<unset>}"
  echo "--- ompi_info pml/osc components ---"
  ompi_info --param pml all --level 9 2>/dev/null | grep -E 'MCA pml:' | head -5 || true
  ompi_info --param osc all --level 9 2>/dev/null | grep -E 'MCA osc:' | head -5 || true
} 2>&1
echo "=== end env diagnostics ==="
echo ""

# Ensure rocshmem4py is installed
if ! python3 -c "import rocshmem4py" 2>/dev/null; then
  echo "Installing rocshmem4py from $PYTHON_SRC_DIR ..."
  pip install -e "$PYTHON_SRC_DIR" || { echo "pip install failed"; exit 1; }
fi

# TEMP: remove before merge -- linkage of the just-built extension
echo "=== TEMP _rocshmem4py.so linkage (remove before merge) ==="
{
  _rocshmem4py_so=$(python3 -c 'import _rocshmem4py; print(_rocshmem4py.__file__)' 2>/dev/null) || true
  if [ -n "$_rocshmem4py_so" ]; then
    echo "$_rocshmem4py_so"
    ldd "$_rocshmem4py_so" 2>&1 | grep -E 'mpi|ucx' || true
  else
    echo "(could not locate _rocshmem4py extension)"
  fi
  # Avoid `import mpi4py.MPI` here -- it calls MPI_Init outside mpirun and
  # spits a harmless but confusing "rendezvous file" error on stderr.
  python3 -c '
import mpi4py, os, glob
print("mpi4py module:", mpi4py.__file__)
ext = glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), "MPI*.so"))
print("mpi4py MPI ext:", ext[0] if ext else "<not found>")
' 2>/dev/null || true
  ext=$(python3 -c "import mpi4py, os, glob; print(glob.glob(os.path.join(os.path.dirname(mpi4py.__file__), 'MPI*.so'))[0])" 2>/dev/null) || true
  if [ -n "$ext" ]; then
    ldd "$ext" 2>&1 | grep -E 'mpi|ucx' || true
  fi
} 2>&1
echo "=== end linkage diagnostics ==="
echo ""

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
  *)
    echo "Unknown test type: $TEST"
    echo "Valid types: all, basic, collective"
    exit 1
    ;;
esac

if [ "$DRIVER_RETURN_STATUS" -eq 0 ]; then
  echo "PYTHON TESTS PASSED"
else
  echo "PYTHON TESTS FAILED:$FAILED_LIST"
fi
exit $DRIVER_RETURN_STATUS
