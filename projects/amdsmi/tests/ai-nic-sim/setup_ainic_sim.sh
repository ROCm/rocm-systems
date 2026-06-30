#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
#
# setup_ainic_sim.sh — set up AI NIC simulation for manual testing.
#
# Creates a fake sysfs tree under /tmp/ainic-sim-<PID>, starts the NIC
# counter simulator, then prints the SMI_NIC_SYSFS_ROOT export command.
#
# Run this in a SEPARATE terminal, then run your test in another terminal:
#
#   Terminal 1:
#       cd projects/amdsmi/tests/ai-nic-sim
#       ./setup_ainic_sim.sh
#       # copy the  export SMI_NIC_SYSFS_ROOT=...  line it prints
#
#   Terminal 2:
#       export SMI_NIC_SYSFS_ROOT=<path from Terminal 1>
#       SMI_NIC_SYSFS_ROOT=... /path/to/amd_smi_ainic_info
#       # or run pytest:
#       pytest projects/amdsmi/tests/ai-nic-sim/

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SIM_ROOT="/tmp/ainic-sim-$$"

cleanup() {
    echo ""
    echo "[setup_ainic_sim] Stopping simulator and cleaning up ..."
    if [[ -n "${SIM_PID:-}" ]] && kill -0 "$SIM_PID" 2>/dev/null; then
        kill "$SIM_PID"
        wait "$SIM_PID" 2>/dev/null || true
    fi
    rm -rf "$SIM_ROOT"
    echo "[setup_ainic_sim] Done."
}
trap cleanup EXIT INT TERM

echo "[setup_ainic_sim] Creating fake sysfs under $SIM_ROOT ..."
HW_COUNTERS_DIR=$(python3 "$SCRIPT_DIR/fake_sysfs.py" "$SIM_ROOT")
echo "[setup_ainic_sim] hw_counters dir: $HW_COUNTERS_DIR"

echo "[setup_ainic_sim] Starting NIC counter simulator ..."
python3 "$SCRIPT_DIR/nic_simulator.py" "$HW_COUNTERS_DIR" &
SIM_PID=$!
echo "[setup_ainic_sim] Simulator PID: $SIM_PID"

echo ""
echo "============================================================"
echo "  AI NIC simulation is ACTIVE."
echo ""
echo "  In your test terminal, run:"
echo ""
echo "    export SMI_NIC_SYSFS_ROOT=$SIM_ROOT"
echo ""
echo "  Then run a test, e.g.:"
echo ""
echo "    /path/to/amd_smi_ainic_info"
echo "    pytest projects/amdsmi/tests/ai-nic-sim/"
echo "============================================================"
echo ""
echo "[setup_ainic_sim] Press Ctrl-C to stop."

while kill -0 "$SIM_PID" 2>/dev/null; do
    sleep 2
done

echo "[setup_ainic_sim] Simulator exited unexpectedly."
exit 1
