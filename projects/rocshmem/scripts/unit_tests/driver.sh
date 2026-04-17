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

#!/bin/bash

# Function to display help information
function display_help {
    echo "Usage:"
    echo "  $0 binary_name all                     # Runs all standard tests"
    echo "  $0 binary_name custom <ranks> <filter> # Runs custom test configuration"
    echo
    echo "Arguments:"
    echo "  binary_name: Name of the binary to run."
    echo "  all: Executes predefined test configurations."
    echo "  custom: Executes a test with custom MPI ranks and GTest filter."
    echo "  ranks: Number of MPI ranks (required for custom mode)."
    echo "  filter: GTest filter string (required for custom mode)."
    echo
}

# Validate number of arguments for each mode
if [[ "$#" -lt 2 ]] ||
   { [[ "$2" == "all" ]] && [[ "$#" -ne 2 ]]; } ||
   { [[ "$2" == "custom" ]] && [[ "$#" -ne 4 ]]; }; then
    display_help
    exit 1
fi

# Get the number of GPUs on the node
if command -v amd-smi >/dev/null && amd-smi version 2>&1 >/dev/null
then
  NUM_GPUS=${NUM_GPUS:-$(amd-smi list | grep GPU | wc -l)}
elif command -v rocm-smi >/dev/null && rocm-smi --version 2>&1 >/dev/null
then
  NUM_GPUS=${NUM_GPUS:-$(rocm-smi --showserial | grep GPU | wc -l)}
fi
NUM_GPUS=${NUM_GPUS:-0}
NUM_GPUS=$(($NUM_GPUS > 0? $NUM_GPUS: 8))

driver_return_status=0
binary_name=$1
mode=$2
timestamp=$(date "+%Y-%m-%d-%H:%M:%S")
log_file="unit_tests_${timestamp}.log"
mpi_timeout=$((20 * 60)) # 20 minutes in seconds

###############################################################################
# Launcher detection helpers
###############################################################################

# Returns the launcher family: ompi_like, slurm, mpich_like, flux, torch.
# For mpirun/mpiexec the brand is inferred from --version output.
launcher_family() {
  local launcher="$1"
  case "$launcher" in
    prterun|paratool) echo "ompi_like";  return ;;
    srun)             echo "slurm";      return ;;
    flux)             echo "flux";       return ;;
    torchrun)         echo "torch";      return ;;
  esac
  local ver
  ver=$("$launcher" --version 2>&1 || true)
  if echo "$ver" | grep -qi "open.mpi\|openmpi"; then echo "ompi_like"
  elif echo "$ver" | grep -qi "mpich\|hydra"; then echo "mpich_like"
  else echo "mpich_like"; fi
}

# Detect which MPI launcher to use.
detect_launcher() {
  if [[ -n "$LAUNCHER" ]]; then echo "$LAUNCHER"; return; fi
  if [[ -n "$SLURM_JOB_ID" ]]; then echo "srun"; return; fi
  if [[ -n "$FLUX_JOB_ID"  ]]; then echo "flux"; return; fi
  if [[ -n "$PBS_JOBID"    ]]; then
    command -v mpiexec &>/dev/null && echo "mpiexec" || echo "mpirun"
    return
  fi
  for candidate in mpirun mpiexec prterun paratool torchrun; do
    command -v "$candidate" &>/dev/null && { echo "$candidate"; return; }
  done
  echo "mpirun"
}

# Function to execute the test under the detected launcher
function run_mpirun {
    local np=$1
    local gtest_filter=$2

    local launcher family
    launcher=$(detect_launcher)
    family=$(launcher_family "$launcher")

    # Env preamble: POSIX `env VAR=val` captures values at call time,
    # no side effects on the parent shell between test runs.
    local -a _env
    _env=( env "UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=16384" )
    [[ -n "$ROCSHMEM_TEST_UUID" ]] && _env+=( "ROCSHMEM_TEST_UUID=$ROCSHMEM_TEST_UUID" )

    local -a cmd
    case "$family" in
      ompi_like)
        cmd=( "${_env[@]}" "$launcher" -n "$np" --timeout "$mpi_timeout" )
        ;;
      slurm)
        cmd=( "${_env[@]}" "$launcher" -n "$np" --time "$((mpi_timeout/60+1))" )
        ;;
      flux)
        cmd=( "${_env[@]}" "$launcher" run -n "$np" --time="${mpi_timeout}s" )
        ;;
      torch)
        local nproc="${TORCHRUN_NPROC_PER_NODE:-1}"
        local nnodes=$(( (np + nproc - 1) / nproc ))
        cmd=( "${_env[@]}" "$launcher" --nproc-per-node "$nproc" --nnodes "$nnodes" )
        ;;
      mpich_like|*)
        cmd=( "${_env[@]}" "$launcher" -n "$np" )
        ;;
    esac

    cmd+=( "$binary_name" "--gtest_filter=$gtest_filter" )

    # Log the full shell-quoted command (including env VAR=val prefix) so it
    # is visible in the log and can be copy-pasted for manual reproduction.
    local cmd_str
    cmd_str="$(printf '%q ' "${cmd[@]}")"
    echo "$cmd_str"
    "${cmd[@]}" >> "$log_file" 2>&1

    # Test if the launcher failed
    if [ $? -ne 0 ]
    then
        echo "FAILED: $cmd_str" >&2
        cat $log_file
        driver_return_status=1
    fi
}

if [ -n "$(rocminfo | grep gfx1201)" ];
then
  echo "Unit tests disabled in gfx1201"
  echo "See AIROCSHMEM-393"
  # Create empty log files for jenkins to be happy
  >$log_file
  exit
fi

# Processing modes
case $mode in
    all)
        test_with_two_pes="IPCImplSimpleCoarseTestFixture/*:IPCImplSimpleFineTestFixture/*:IPCImplTiledFineTestFixture/*:DegenerateTiledFine.*"
        test_with_two_pes+=":SdmaSimpleCoarse/*:SdmaSimpleFine/*:SdmaTiledFine/*"
        if [ $NUM_GPUS -ge 4 ]
        then
          run_mpirun 4 "-$test_with_two_pes"
        fi
        #run_mpirun 2 "$test_with_two_pes"
        ;;
    custom)
        # Check if ranks is a positive integer
        if [[ "$3" -le 1 ]]; then
            echo "Error: 'ranks' must be a positive integer."
            display_help
            exit 1
        fi
        run_mpirun $3 $4
        ;;
    *)
        echo "Error: Invalid mode '$mode'." | tee -a "$log_file"
        display_help
        exit 1
        ;;
esac

echo "Tests Completed"
echo "log file: '$log_file'"
exit $driver_return_status
