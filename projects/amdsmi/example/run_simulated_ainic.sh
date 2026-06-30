#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# run_simulated_ainic.sh — manual developer test for AI NIC simulation.
#
# Creates a fake sysfs tree, starts the NIC simulator in the background,
# then runs amd_smi_ainic_info once so you can see the output.
#
# Usage:
#   ./run_simulated_ainic.sh [path/to/amd_smi_ainic_info]
#
# If the binary path is not given, the script looks for amd_smi_ainic_info
# in the PATH and in common build directories relative to the repo root.
#
# Requirements: python3 must be available.
#
# Environment variables (optional):
#   NIC_SIM_INTERVAL  — simulator update interval in seconds (default: 0.1)

set -euo pipefail

# ---------------------------------------------------------------------------
# Resolve script and repo locations
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# This script lives at projects/amdsmi/example/; simulation scripts are at
# projects/amdsmi/tests/ai-nic-sim/
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
SIM_DIR="${REPO_ROOT}/projects/amdsmi/tests/ai-nic-sim"

FAKE_SYSFS_PY="${SIM_DIR}/fake_sysfs.py"
NIC_SIM_PY="${SIM_DIR}/nic_simulator.py"

# ---------------------------------------------------------------------------
# Locate the amd_smi_ainic_info binary
# ---------------------------------------------------------------------------
if [[ $# -ge 1 ]]; then
    BINARY="$1"
else
    BINARY=""
    for candidate in \
        "$(command -v amd_smi_ainic_info 2>/dev/null || true)" \
        "${REPO_ROOT}/build/projects/amdsmi/example/amd_smi_ainic_info" \
        "${REPO_ROOT}/build-release/projects/amdsmi/example/amd_smi_ainic_info" \
        "${SCRIPT_DIR}/build/amd_smi_ainic_info"
    do
        if [[ -x "${candidate}" ]]; then
            BINARY="${candidate}"
            break
        fi
    done

    if [[ -z "${BINARY}" ]]; then
        echo "ERROR: amd_smi_ainic_info not found."
        echo "  Build it with CMake first, then either:"
        echo "    $0 /path/to/amd_smi_ainic_info"
        echo "  or put it in PATH."
        exit 1
    fi
fi

if [[ ! -x "${BINARY}" ]]; then
    echo "ERROR: '${BINARY}' is not an executable file."
    exit 1
fi

echo "Binary : ${BINARY}"

# ---------------------------------------------------------------------------
# Validate Python helpers
# ---------------------------------------------------------------------------
for f in "${FAKE_SYSFS_PY}" "${NIC_SIM_PY}"; do
    if [[ ! -f "${f}" ]]; then
        echo "ERROR: Required Python script not found: ${f}"
        exit 1
    fi
done

# ---------------------------------------------------------------------------
# Create the fake sysfs tree
# ---------------------------------------------------------------------------
FAKE_ROOT="$(mktemp -d /tmp/ainic-sim-sysfs.XXXXXX)"
echo "Fake sysfs : ${FAKE_ROOT}"
python3 "${FAKE_SYSFS_PY}" "${FAKE_ROOT}" > /dev/null

HW_COUNTERS_DIR="${FAKE_ROOT}/sys/devices/pci0000:e0/0000:e2:00.0/0000:e2:00.1/infiniband/rocep226s0/ports/1/hw_counters"

# ---------------------------------------------------------------------------
# Start the NIC simulator in the background
# ---------------------------------------------------------------------------
NIC_SIM_INTERVAL="${NIC_SIM_INTERVAL:-0.1}"
echo "Starting NIC simulator (interval=${NIC_SIM_INTERVAL}s) ..."
python3 "${NIC_SIM_PY}" "${HW_COUNTERS_DIR}" --interval "${NIC_SIM_INTERVAL}" &
SIM_PID=$!

cleanup() {
    echo ""
    echo "Stopping NIC simulator (pid ${SIM_PID}) ..."
    kill "${SIM_PID}" 2>/dev/null || true
    wait "${SIM_PID}" 2>/dev/null || true
    echo "Removing fake sysfs: ${FAKE_ROOT}"
    rm -rf "${FAKE_ROOT}"
}
trap cleanup EXIT

# Give the simulator a moment to write the first tick
sleep 0.3

# ---------------------------------------------------------------------------
# Run amd_smi_ainic_info with the fake sysfs root
# ---------------------------------------------------------------------------
echo ""
echo "Running: SMI_NIC_SYSFS_ROOT=${FAKE_ROOT} ${BINARY}"
echo "============================================================"
SMI_NIC_SYSFS_ROOT="${FAKE_ROOT}" "${BINARY}"
echo "============================================================"
