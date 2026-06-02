#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RCCL_INSTALL_DIR="${RCCL_INSTALL_DIR:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"
export OMPI_INSTALL_DIR="${OMPI_INSTALL_DIR:-${MPI_PATH:-/opt/ompi}}"
RCCL_BUILD_DIR="${RCCL_BUILD_DIR:-${RCCL_INSTALL_DIR}/build/debug}"

# Check if rccl-tests is present in the parent directory. If not, pull it in.
RCCL_TESTS_SRC="${RCCL_INSTALL_DIR}/../rccl-tests"
if [ ! -d "${RCCL_TESTS_SRC}" ]; then
    GIT_ROOT="$(git -C "${RCCL_INSTALL_DIR}" rev-parse --show-toplevel 2>/dev/null || true)"
    [ -n "${GIT_ROOT}" ] && git -C "${GIT_ROOT}" sparse-checkout add projects/rccl-tests
fi
RCCL_TESTS_SRC="$(cd "${RCCL_TESTS_SRC}" && pwd)"
export RCCL_TESTS_DIR="${RCCL_TESTS_SRC}"

# Build rccl-tests with MPI.
(cd "${RCCL_TESTS_SRC}" && ./install.sh --mpi --mpi_home="${OMPI_INSTALL_DIR}" \
    --rccl_home="${RCCL_BUILD_DIR}" --gpu_targets "${GPU_TARGETS:-native}")

# Build the three plugin shared libs that the pytest suite expects on disk.
for d in ext-profiler/example ext-tuner/example ext-profiler/inspector; do
    make -C "${RCCL_INSTALL_DIR}/${d}" clean
    make -C "${RCCL_INSTALL_DIR}/${d}"
done

# Create and activate venv for the pytest dependencies.
cd "${SCRIPT_DIR}"
python3 -m venv venv
source venv/bin/activate
pip install -q -r requirements.txt

# Run ext-plugin pytest suite
pytest -v --cache-clear
