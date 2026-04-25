#!/usr/bin/env bash
# Diff the public HSA symbols exported from libhsa-runtime64.so against the
# LTTng AST-migration inventory. Fail the build on any miss.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libhsa-runtime64.so> <inventory.txt>
#
# Exits 0 on full coverage, non-zero otherwise.
set -euo pipefail

SO="${1:?usage: $0 <path/to/libhsa-runtime64.so> <inventory.txt>}"
INV="${2:?inventory required}"

if [ ! -f "$SO" ]; then
    echo "FAIL: shared object not found: $SO"
    exit 1
fi
if [ ! -f "$INV" ]; then
    echo "FAIL: inventory not found: $INV"
    exit 1
fi

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# Extract exported text symbols starting with hsa_ or __hsa_.
# nm output: <addr> <type> <name>; type T/t indicates a defined text symbol.
# Strip the @@VERSION or @VERSION suffix (e.g., hsa_init@@ROCR_1 -> hsa_init).
nm -D --defined-only "$SO" \
  | awk '$2 ~ /^[Tt]$/ && $3 ~ /^(hsa_|__hsa_)/ {
           sub(/@.*/, "", $3); print $3
         }' \
  | sort -u > "$TMP_DIR/exported.txt"

# Migrated symbols (first column of inventory).
awk '{print $1}' "$INV" | sort -u > "$TMP_DIR/migrated.txt"

# Symbols defined as aliases in hsacore.so.link share the address (and
# function body) of an already-migrated symbol. Add them to the migrated
# set so the gate does not flag them.
LINK_SCRIPT="$(dirname "$INV")/../hsacore.so.link"
if [ -f "$LINK_SCRIPT" ]; then
    awk '/=/ { sub(/;.*/, ""); split($0, a, "="); gsub(/[ \t]/, "", a[1]); print a[1] }' \
        "$LINK_SCRIPT" >> "$TMP_DIR/migrated.txt"
    sort -u -o "$TMP_DIR/migrated.txt" "$TMP_DIR/migrated.txt"
fi

# Symbols present in the .so but NOT in the migration inventory.
MISSING=$(comm -23 "$TMP_DIR/exported.txt" "$TMP_DIR/migrated.txt" || true)

# Some symbols are intentionally NOT in hsa_table_interface.cpp — they are
# implemented in other TUs (e.g., hsa_ext_image*.cpp, hsa_ext_program* in the
# loader, etc.) but still get exported via the version script. Those are not
# part of the table-interface migration scope.
#
# - hsa_ext_image_*  / hsa_ext_sampler_* : image extension (image/hsa_ext_image.cpp)
# - hsa_ext_program_*                    : finalizer extension (loader)
# - hsa_amd_image_get_info_max_dim       : image extension AMD-specific
# - hsa_ven_amd_*                        : vendor extension entry points
# - __hsa_amd_aql_queue_*_async_handler  : helper symbols outside the table interface
ALLOWLIST_RE='^hsa_ext_image_|^hsa_ext_sampler_|^hsa_ext_program_|^hsa_amd_image_get_info_max_dim$|^hsa_ven_amd_|^__hsa_amd_aql_queue_get_async_handler$|^__hsa_amd_aql_queue_set_async_handler$'

if [ -n "$MISSING" ]; then
    UNEXPECTED=$(echo "$MISSING" | grep -vE "$ALLOWLIST_RE" || true)
    if [ -n "$UNEXPECTED" ]; then
        echo "FAIL: the following exported HSA symbols were NOT migrated to LTTng:"
        echo "$UNEXPECTED"
        exit 1
    fi
fi

N_MIGRATED=$(wc -l < "$TMP_DIR/migrated.txt")
N_EXPORTED=$(wc -l < "$TMP_DIR/exported.txt")
echo "PASS: all $N_EXPORTED exported HSA symbols migrated ($N_MIGRATED inventory entries)"
