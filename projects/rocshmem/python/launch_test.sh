#!/bin/bash
################################################################################
# Launch script for rocshmem4py tests
#
# The RO backend requires MPI-launched processes.  This script uses mpirun
# by default and init_with_torch() inside the test for coordination
# (matching the Triton-distributed pattern).
#
# Usage:
#   ./launch_test.sh -n <num_procs> <script_or_cmd>
#
# Examples:
#   ./launch_test.sh -n 2 -c "pytest tests/ -v"
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set -e

print_msg()   { echo "[rocshmem4py] $1"; }
print_error() { echo "[ERROR] $1"; }

# Defaults
NUM_PROCS=1
OMPI_DIR="${OMPI_DIR:-/opt/ompi_build/install/ompi}"
ROCSHMEM_BUILD="${ROCSHMEM_BUILD:-$(dirname "$0")/../build}"
HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-536870912}"
UCX_SIGPOOL="${UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS:-16384}"
USE_COMMAND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nprocs)  NUM_PROCS="$2"; shift 2 ;;
        -c|--command) USE_COMMAND=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <script_or_command>"
            echo ""
            echo "Options:"
            echo "  -n, --nprocs N      Number of MPI processes (default: 1)"
            echo "  -c, --command       Treat argument as shell command"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  OMPI_DIR                         UCX-enabled MPI directory"
            echo "  ROCSHMEM_BUILD                   rocSHMEM build directory"
            echo "  ROCSHMEM_HEAP_SIZE               Heap size in bytes (default: 536870912)"
            echo "  ROCSHMEM_USE_TORCH_INIT          1=torch init (default), 0=MPI init"
            echo "  UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS   UCX signal pool size"
            echo ""
            echo "Example:"
            echo "  $0 -n 2 -c 'pytest tests/ -v'"
            exit 0 ;;
        *) SCRIPT_OR_CMD="$@"; break ;;
    esac
done

if [ -z "$SCRIPT_OR_CMD" ]; then
    print_error "No script or command specified"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

# Verify MPI
if [ ! -d "$OMPI_DIR" ]; then
    print_error "UCX-enabled MPI not found at: $OMPI_DIR"
    echo "Set OMPI_DIR to your UCX-enabled MPI installation"
    exit 1
fi
if [ ! -f "$OMPI_DIR/bin/mpirun" ]; then
    print_error "mpirun not found in $OMPI_DIR/bin/"
    exit 1
fi

# Setup environment
export PATH="$OMPI_DIR/bin:$PATH"
export LD_LIBRARY_PATH="$OMPI_DIR/lib:$ROCSHMEM_BUILD:${LD_LIBRARY_PATH:-}"

print_msg "Verifying rocshmem4py installation..."
if ! python3 -c "import rocshmem4py" 2>/dev/null; then
    print_error "Failed to import rocshmem4py. Run: pip install -e ."
    exit 1
fi

print_msg "Environment:"
echo "  MPI:        $OMPI_DIR"
echo "  rocSHMEM:   $ROCSHMEM_BUILD"
echo "  Processes:  $NUM_PROCS"
echo "  Heap size:  $HEAP_SIZE"
echo "  Init mode:  ${ROCSHMEM_USE_TORCH_INIT:-1} (1=torch, 0=mpi)"
echo ""

# Build mpirun command
MPI_ARGS="--allow-run-as-root -n $NUM_PROCS"
MPI_ARGS="$MPI_ARGS -mca pml ucx -mca osc ucx"
MPI_ARGS="$MPI_ARGS -x ROCSHMEM_HEAP_SIZE=$HEAP_SIZE"
MPI_ARGS="$MPI_ARGS -x UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS=$UCX_SIGPOOL"
MPI_ARGS="$MPI_ARGS -x LD_LIBRARY_PATH"
MPI_ARGS="$MPI_ARGS -x WORLD_SIZE=$NUM_PROCS"

if [ "$USE_COMMAND" = true ]; then
    CMD="mpirun $MPI_ARGS $SCRIPT_OR_CMD"
else
    CMD="mpirun $MPI_ARGS python3 $SCRIPT_OR_CMD"
fi

print_msg "Starting test..."
echo ""

set +e
OUTPUT=$(eval "$CMD" 2>&1)
run_ret=$?
set -e

echo "$OUTPUT"
echo ""

if echo "$OUTPUT" | grep -qi "PASSED"; then
    if echo "$OUTPUT" | grep -qi "FAILED\|ERROR.*assert"; then
        print_error "Some tests failed"
        exit 1
    else
        print_msg "All tests passed!"
        exit 0
    fi
fi

if [ $run_ret -eq 0 ]; then
    print_msg "Completed successfully!"
else
    print_error "Failed with exit code $run_ret"
fi

exit $run_ret
