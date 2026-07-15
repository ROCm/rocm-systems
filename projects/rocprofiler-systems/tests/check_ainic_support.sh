#!/usr/bin/env bash
# check_ainic_support.sh
# Usage: check_ainic_support.sh <path/to/librocprof-sys.so>
# Exit 0 = AINIC compiled in, Exit 1 = not compiled in, Exit 2 = library not found.

LIB="${1:-}"

if [[ -z "$LIB" || ! -f "$LIB" ]]; then
    echo "ERROR: librocprof-sys.so not found. Pass the path as the first argument."
    exit 2
fi

echo "Checking: $LIB"

NM_OUT=$(nm -D "$LIB" 2>&1)
NM_RC=$?

if [[ $NM_RC -ne 0 ]]; then
    echo "ERROR: nm -D failed (exit $NM_RC): $NM_OUT"
    exit 2
fi

if echo "$NM_OUT" | grep -q 'amdsmi_get_nic_rdma_port_statistics'; then
    echo "PASS: AI NIC support IS compiled in (amdsmi_get_nic_rdma_port_statistics found in dynamic symbols)"
    exit 0
fi

echo "FAIL: AI NIC support is NOT compiled in"
exit 1
