#!/bin/bash
# rebuild-and-reload.sh
#
# Incrementally rebuild amdgpu.ko from the existing DKMS build directory and
# reload it. Much faster than a full dkms build for iterative development.
#
# Prerequisites:
#   - At least one full `dkms build` has been run to establish the build dir.
#   - Run inside the guest VM.
#
# Usage:
#   ./rebuild-and-reload.sh [DKMS_VER]
#
# Example:
#   ./rebuild-and-reload.sh 6.19.4-2337710.24.04

set -euo pipefail

FORCE=${1:-}
if [ "${FORCE}" = "--force" ]; then
    shift
fi

KERNEL=$(uname -r)
DKMS_VER=${1:-}

# Auto-detect the installed amdgpu-dkms version if not specified.
if [ -z "${DKMS_VER}" ]; then
    DKMS_VER=$(ls /var/lib/dkms/amdgpu/ 2>/dev/null | head -1)
    if [ -z "${DKMS_VER}" ]; then
        echo "ERROR: No amdgpu DKMS build found in /var/lib/dkms/amdgpu/."
        echo "       Run: sudo dkms build amdgpu/<ver> -k ${KERNEL}"
        exit 1
    fi
fi

SRC_DIR="/usr/src/amdgpu-${DKMS_VER}"
MODULE="${SRC_DIR}/amd/amdgpu/amdgpu.ko"
if [ ! -d "${SRC_DIR}" ]; then
    echo "ERROR: Source dir not found: ${SRC_DIR}"
    exit 1
fi

echo "==> Kernel:     ${KERNEL}"
echo "==> DKMS ver:   ${DKMS_VER}"
echo "==> Source dir: ${SRC_DIR}"
echo ""

echo "==> Ensuring -Wno-unused-variable in Makefile..."
if ! grep -q "Wno-unused-variable" "${SRC_DIR}/amd/amdgpu/Makefile"; then
    sudo sed -i '1s/^/ccflags-y += -Wno-unused-variable\n/' "${SRC_DIR}/amd/amdgpu/Makefile"
fi

CONFIG_H="${SRC_DIR}/amd/dkms/config/config.h"
if [ ! -f "${CONFIG_H}" ]; then
    echo "==> Running pre-build.sh to generate config.h (first time only)..."
    sudo bash -c "cd '${SRC_DIR}' && amd/dkms/pre-build.sh '${KERNEL}' '${SRC_DIR}' '${SRC_DIR}' '$(which gcc)'"
fi

echo "==> Building amdgpu.ko (incremental)..."
sudo make -C "/lib/modules/${KERNEL}/build" \
    M="${SRC_DIR}" \
    src="${SRC_DIR}" \
    modules \
    -j"$(nproc)"

echo ""
echo "==> Unloading amdgpu..."
sudo rmmod amdgpu 2>/dev/null || true

echo "==> Loading amdgpu.ko..."
sudo insmod "${MODULE}"

echo ""
echo "==> Done. dmesg tail:"
dmesg | tail -20
