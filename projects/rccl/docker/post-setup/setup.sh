#!/bin/bash
#
# Post-setup: build RCCL-Tests with MPI from /opt/shared/ompi.
#
# Shared/non-shared detection: checks if the build output already exists.
#   - Shared FS:     first container builds, others find the binary → skip
#   - Non-shared FS: each container builds independently
#   - Rebuild:       FORCE_POST_SETUP=1 clears markers, script re-checks output
#
# Invoked automatically by the entrypoint's post-setup hook:
#   python3 -m mnctl --launch-all --post-setup post-setup/

set -eo pipefail

WORKDIR="${WORKDIR:-/workspace}"
RCCL_INSTALL_PREFIX="${RCCL_INSTALL_PREFIX:-${WORKDIR}/rocm-systems/projects/rccl/install}"
GPU_TARGETS="${GPU_TARGETS:-gfx942}"
MPI_PREFIX="/opt/shared/ompi"
SRC="${WORKDIR}/rocm-systems/projects/rccl-tests"
BINARY="${SRC}/build/all_reduce_perf"

# --- Shared/non-shared detection: skip if output already exists ---
if [[ -x "${BINARY}" ]]; then
    echo "RCCL-Tests: already built at ${BINARY}"
    exit 0
fi

if [[ ! -d "${MPI_PREFIX}/lib" ]]; then
    echo "ERROR: MPI not found at ${MPI_PREFIX}"
    echo "  Run --setup-deps first to build UCX/OpenMPI into the shared directory."
    exit 1
fi

if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: RCCL-Tests source not found at ${SRC}"
    exit 1
fi

echo "Building RCCL-Tests with MPI from ${MPI_PREFIX}"

rm -rf "${SRC}/build"
mkdir -p "${SRC}/build"
cd "${SRC}/build"

cmake -DCMAKE_BUILD_TYPE=Release \
    -DUSE_MPI=ON \
    -DCMAKE_PREFIX_PATH="${RCCL_INSTALL_PREFIX};${MPI_PREFIX}" \
    -DGPU_TARGETS="${GPU_TARGETS}" ..

make -j"$(nproc)"

echo "RCCL-Tests built in ${SRC}/build"
