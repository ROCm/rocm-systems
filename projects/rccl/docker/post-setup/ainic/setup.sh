#!/bin/bash
#
# AINIC driver installation for Ubuntu-based containers.
#
# Prerequisites:
#   Mount the AINIC driver source into the container:
#     --volume /path/to/drivers-linux:/opt/nic-drivers:ro
#
# The driver source directory should contain:
#   - setup_libs.sh
#   - rdma-core/  (with CMakeLists.txt)
#

set -euo pipefail

DRIVER_SRC="${NIC_DRIVER_DIR:-/opt/nic-drivers}"

if [[ ! -d "${DRIVER_SRC}" ]]; then
    echo "ERROR: AINIC driver source not found at ${DRIVER_SRC}"
    echo "Mount it with: --volume /host/path/drivers-linux:${DRIVER_SRC}:ro"
    exit 1
fi

echo "Installing AINIC driver from ${DRIVER_SRC}..."

apt-get update -qq
DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    python3-dev \
    cython3 \
    libsystemd-dev \
    libudev-dev \
    libnl-3-dev \
    libnl-route-3-dev \
    python3-docutils \
    ninja-build \
    rdma-core \
    ibverbs-utils \
    infiniband-diags \
    cmake

WORK_DIR=$(mktemp -d /tmp/ainic-build.XXXXXX)
cp -a "${DRIVER_SRC}/." "${WORK_DIR}/"
cd "${WORK_DIR}"

if [[ -f setup_libs.sh ]]; then
    echo "Running setup_libs.sh..."
    bash ./setup_libs.sh
    sleep 1
fi

if [[ -d rdma-core ]]; then
    echo "Building rdma-core..."
    mkdir -p rdma-core/build
    cd rdma-core/build
    cmake -GNinja \
        -DCMAKE_INSTALL_PREFIX:PATH=/usr \
        -DNO_PYVERBS=1 \
        -DNO_MAN_PAGES=1 \
        ${EXTRA_CMAKE_FLAGS:-} \
        ..
    sleep 1
    ninja install
    sleep 1
fi

echo "Verifying IB devices..."
ibv_devices || echo "WARN: ibv_devices failed (may need host kernel modules)"

rm -rf "${WORK_DIR}"

echo "AINIC driver installation complete."
