#!/usr/bin/env bash
# CTest wrapper: run CLI integration tests inside a QEMU VM via SSH.
#
# Required env vars (set via CTest -D or ctest --build-and-test):
#   SDXI_VM_SSH   — SSH command prefix, e.g.
#                   "ssh -o StrictHostKeyChecking=no -p 2236 -i ~/.ssh/id_rsa ubuntu@localhost"
#
# Optional:
#   SDXI_VM_CLI   — path to sdxi-cli binary inside the VM (default: ~/sdxi-cli/sdxi-cli)
#   SDXI_VM_TESTS — path to run_tests.sh inside the VM   (default: ~/tests-cli/run_tests.sh)
#
# If SDXI_VM_SSH is unset this script exits 77 (CTest SKIP).

set -euo pipefail

if [[ -z "${SDXI_VM_SSH:-}" ]]; then
    echo "SKIP: SDXI_VM_SSH not set — VM CLI tests require a running QEMU VM"
    exit 77
fi

VM_CLI="${SDXI_VM_CLI:-~/sdxi-cli/sdxi-cli}"
VM_TESTS="${SDXI_VM_TESTS:-~/tests-cli/run_tests.sh}"

exec ${SDXI_VM_SSH} "SDXI_CLI=${VM_CLI} bash ${VM_TESTS} -v"
