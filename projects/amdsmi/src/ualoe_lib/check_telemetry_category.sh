#!/bin/bash
# SPDX-License-Identifier: GPL-2.0
# Drift check: verifies every IFOE_TELEM_CAT_* define in ifoe_telemetry_hostapp.h
# has a corresponding UALOE_TELEMETRY_CATEGORY_* entry in ualoe_lib.h.
# Fails the build if any FW category is missing from the hand-written enum.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INC_DIR="${SCRIPT_DIR}/../../include/ualoe_lib"
HOSTAPP="${INC_DIR}/ifoe_telemetry_hostapp.h"
UALOE_LIB="${INC_DIR}/ualoe_lib.h"

# Extract FW category names in numeric order.
# printf "%d" in the shell handles 0x... hex literals portably (no awk strtonum needed).
fw_cats=$(
    grep -E "^#define[[:space:]]+IFOE_TELEM_CAT_[A-Z]" "${HOSTAPP}" \
    | awk '{print $3, $2}' \
    | while IFS=' ' read -r hex name; do printf '%d %s\n' "$hex" "$name"; done \
    | sort -n \
    | awk '{sub(/IFOE_TELEM_CAT_/, "", $2); print $2}'
)

# Extract hand-written enum names, strip prefix, exclude _MAX sentinel.
# Skip #define lines so macros like UALOE_TELEMETRY_CATEGORY_MASK(cat) are not matched.
ualoe_cats=$(
    grep -v '^#define' "${UALOE_LIB}" \
    | grep -oE 'UALOE_TELEMETRY_CATEGORY_[A-Z_]+' \
    | grep -v '_MAX$' \
    | sed 's/UALOE_TELEMETRY_CATEGORY_//' \
    | sort -u
)

rc=0
while IFS= read -r cat; do
    [ -z "$cat" ] && continue
    if ! printf '%s\n' "$ualoe_cats" | grep -qx "$cat"; then
        if [ "$rc" -eq 0 ]; then
            echo "ERROR: ualoe_telemetry_category_e is out of sync with ifoe_telemetry_hostapp.h" >&2
            echo "Add the following to the enum in ualoe_lib.h:" >&2
        fi
        echo "  UALOE_TELEMETRY_CATEGORY_${cat}," >&2
        rc=1
    fi
done << CATS
$fw_cats
CATS

while IFS= read -r cat; do
    [ -z "$cat" ] && continue
    if ! printf '%s\n' "$fw_cats" | grep -qx "$cat"; then
        if [ "$rc" -eq 0 ]; then
            echo "ERROR: ualoe_telemetry_category_e is out of sync with ifoe_telemetry_hostapp.h" >&2
            echo "Remove the following from the enum in ualoe_lib.h:" >&2
        fi
        echo "  UALOE_TELEMETRY_CATEGORY_${cat}," >&2
        rc=1
    fi
done << UALOE_CATS
$ualoe_cats
UALOE_CATS

[ "$rc" -ne 0 ] && exit 1

fw_count=$(printf '%s\n' "$fw_cats" | awk 'NF' | wc -l | tr -d ' ')
echo "telemetry category sync OK (${fw_count} FW categories match enum)" >&2
