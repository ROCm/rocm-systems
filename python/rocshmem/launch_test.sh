#!/bin/bash
################################################################################
# Launch script for rocshmem4py tests
#
# Supports two launchers:
#   * mpirun  (default, required for the RO backend)
#   * torchrun (for IPC / GDA backends that do not need MPI)
#
# Tests call rocshmem4py.init_with_torch() by default, so both launchers
# work; the launcher choice only controls how processes are spawned and
# whether an MPI runtime is required.
#
# Usage:
#   ./launch_test.sh [-l mpirun|torchrun] -n <num_procs> <script_or_cmd>
#
# Examples:
#   # RO backend (default)
#   ./launch_test.sh -n 2 -c "pytest tests/ -v"
#
#   # IPC / GDA backend
#   ./launch_test.sh -l torchrun -n 2 -c "pytest tests/ -v"
#
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
################################################################################

set -e

print_msg()   { echo "[rocshmem4py] $1"; }
print_error() { echo "[ERROR] $1"; }

# Defaults
LAUNCHER="${ROCSHMEM_LAUNCHER:-mpirun}"
NUM_PROCS=1
OMPI_DIR="${OMPI_DIR:-/opt/ompi_build/install/ompi}"
ROCSHMEM_BUILD="${ROCSHMEM_BUILD:-$(dirname "$0")/../build}"
HEAP_SIZE="${ROCSHMEM_HEAP_SIZE:-536870912}"
UCX_SIGPOOL="${UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS:-16384}"
USE_COMMAND=false

while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--nprocs)   NUM_PROCS="$2"; shift 2 ;;
        -l|--launcher) LAUNCHER="$2"; shift 2 ;;
        -c|--command)  USE_COMMAND=true; shift ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS] <script_or_command>"
            echo ""
            echo "Options:"
            echo "  -l, --launcher L    mpirun (default, RO) | torchrun (IPC/GDA)"
            echo "  -n, --nprocs   N    Number of processes (default: 1)"
            echo "  -c, --command       Treat argument as shell command"
            echo "  -h, --help          Show this help"
            echo ""
            echo "Environment Variables:"
            echo "  ROCSHMEM_LAUNCHER                Default launcher (mpirun|torchrun)"
            echo "  OMPI_DIR                         UCX-enabled MPI directory (mpirun only)"
            echo "  ROCSHMEM_BUILD                   rocSHMEM build directory"
            echo "  ROCSHMEM_HEAP_SIZE               Heap size in bytes (default: 536870912)"
            echo "  ROCSHMEM_MASTER_ADDR             Optional torch.dist master address"
            echo "  ROCSHMEM_MASTER_PORT             Optional torch.dist master port"
            echo "  UCX_ROCM_IPC_SIGPOOL_MAX_ELEMS   UCX signal pool size (mpirun only)"
            echo ""
            echo "Examples:"
            echo "  $0 -n 2 -c 'pytest tests/ -v'                   # RO via mpirun"
            echo "  $0 -l torchrun -n 2 -c 'pytest tests/ -v'       # IPC / GDA via torchrun"
            exit 0 ;;
        *) SCRIPT_OR_CMD="$@"; break ;;
    esac
done

if [ -z "$SCRIPT_OR_CMD" ]; then
    print_error "No script or command specified"
    echo "Run '$0 --help' for usage information"
    exit 1
fi

if [ "$LAUNCHER" != "mpirun" ] && [ "$LAUNCHER" != "torchrun" ]; then
    print_error "Unknown launcher: $LAUNCHER (expected mpirun or torchrun)"
    exit 1
fi

# Base environment
export LD_LIBRARY_PATH="$ROCSHMEM_BUILD:${LD_LIBRARY_PATH:-}"

print_msg "Verifying rocshmem4py installation..."
if ! python3 -c "import rocshmem4py" 2>/dev/null; then
    print_error "Failed to import rocshmem4py. Run: pip install -e ."
    exit 1
fi

cleanup() { kill -- -$$ 2>/dev/null; }
trap cleanup INT TERM

if [ "$LAUNCHER" = "mpirun" ]; then
    if [ ! -d "$OMPI_DIR" ]; then
        print_error "UCX-enabled MPI not found at: $OMPI_DIR"
        echo "Set OMPI_DIR to your UCX-enabled MPI installation,"
        echo "or use '-l torchrun' for IPC/GDA backends."
        exit 1
    fi
    if [ ! -f "$OMPI_DIR/bin/mpirun" ]; then
        print_error "mpirun not found in $OMPI_DIR/bin/"
        exit 1
    fi

    export PATH="$OMPI_DIR/bin:$PATH"
    export LD_LIBRARY_PATH="$OMPI_DIR/lib:$LD_LIBRARY_PATH"

    print_msg "Environment:"
    echo "  Launcher:   mpirun ($OMPI_DIR)"
    echo "  rocSHMEM:   $ROCSHMEM_BUILD"
    echo "  Processes:  $NUM_PROCS"
    echo "  Heap size:  $HEAP_SIZE"
    echo ""

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
else
    # torchrun path: no MPI requirement
    if ! command -v torchrun >/dev/null 2>&1; then
        print_error "torchrun not found on PATH; install PyTorch or use '-l mpirun'"
        exit 1
    fi

    export ROCSHMEM_HEAP_SIZE="$HEAP_SIZE"

    print_msg "Environment:"
    echo "  Launcher:   torchrun ($(command -v torchrun))"
    echo "  rocSHMEM:   $ROCSHMEM_BUILD"
    echo "  Processes:  $NUM_PROCS"
    echo "  Heap size:  $HEAP_SIZE"
    echo ""

    TR_ARGS="--standalone --nnodes=1 --nproc_per_node=$NUM_PROCS"

    # torchrun spawns python internally, so translate common command prefixes
    # into its expected "script-or-module + args" form.
    if [ "$USE_COMMAND" = true ]; then
        tr_tail="$SCRIPT_OR_CMD"
        tr_tail="${tr_tail#python3 }"
        tr_tail="${tr_tail#python }"
        case "$tr_tail" in
            pytest\ *|pytest)
                tr_tail="-m pytest ${tr_tail#pytest}"
                ;;
        esac
        CMD="torchrun $TR_ARGS $tr_tail"
    else
        CMD="torchrun $TR_ARGS $SCRIPT_OR_CMD"
    fi
fi

print_msg "Starting test..."
echo ""

set +e
eval "$CMD"
run_ret=$?
set -e
echo ""

if [ $run_ret -eq 0 ]; then
    print_msg "All tests passed!"
else
    print_error "Tests failed with exit code $run_ret"
fi

exit $run_ret
