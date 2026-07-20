#!/usr/bin/env bash
# LTTng coverage gate (HSA).
#
# 1. Symbol coverage: diff the public HSA symbols exported from
#    libhsa-runtime64.so against the LTTng migration inventory.
#    Fails the build on any miss.
#
# 2. Curated coverage: require the curated YAML to match the migration
#    inventory exactly, then require every wrapper to retain its curated
#    sentinel, argument snapshots, and curated return/direct-helper path.
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

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# hsa-runtime root is the parent of scripts/.
HSA_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Shared Python tooling (project-agnostic; curated_apis.yaml and the
# migration inventory itself remain per-project in $SCRIPT_DIR).
SHARED_SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../../../../../shared/lttng/scripts" && pwd)"
COVERAGE_CHECK="$SHARED_SCRIPTS_DIR/lttng_coverage_check.py"

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT

# ---------------------------------------------------------------------------
# 1. Symbol-coverage gate
# ---------------------------------------------------------------------------
# Extract exported text symbols starting with hsa_ or __hsa_.
nm -D --defined-only "$SO" \
  | awk '$2 ~ /^[Tt]$/ && $3 ~ /^(hsa_|__hsa_)/ {
           sub(/@.*/, "", $3); print $3
         }' \
  | sort -u > "$TMP_DIR/exported.txt"

# Migrated symbols (first column of inventory).
awk '{print $1}' "$INV" | sort -u > "$TMP_DIR/migrated.txt"

# Aliases declared in hsacore.so.link share the address (and function body)
# of an already-migrated symbol. Add them so the gate does not flag them.
LINK_SCRIPT="$(dirname "$INV")/../hsacore.so.link"
if [ -f "$LINK_SCRIPT" ]; then
    awk '/=/ { sub(/;.*/, ""); split($0, a, "="); gsub(/[ \t]/, "", a[1]); print a[1] }' \
        "$LINK_SCRIPT" >> "$TMP_DIR/migrated.txt"
    sort -u -o "$TMP_DIR/migrated.txt" "$TMP_DIR/migrated.txt"
fi

MISSING=$(comm -23 "$TMP_DIR/exported.txt" "$TMP_DIR/migrated.txt" || true)

# Some symbols are intentionally NOT in hsa_table_interface.cpp — they are
# implemented in other TUs (e.g., hsa_ext_image*.cpp, hsa_ext_program* in the
# loader, etc.) but still get exported via the version script. Those are not
# part of the table-interface migration scope.
ALLOWLIST_RE='^hsa_ext_image_|^hsa_ext_sampler_|^hsa_ext_program_|^hsa_amd_image_get_info_max_dim$|^hsa_ven_amd_|^__hsa_amd_aql_queue_get_async_handler$|^__hsa_amd_aql_queue_set_async_handler$'

if [ -n "$MISSING" ]; then
    UNEXPECTED=$(echo "$MISSING" | grep -vE "$ALLOWLIST_RE" || true)
    if [ -n "$UNEXPECTED" ]; then
        echo "FAIL: the following exported HSA symbols were NOT migrated to LTTng:"
        echo "$UNEXPECTED"
        exit 1
    fi
fi

# ---------------------------------------------------------------------------
# 2. HSA curated-args coverage gate
# ---------------------------------------------------------------------------
CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ ! -f "$CURATED_YAML" ]; then
    echo "FAIL: required curated API inventory not found: $CURATED_YAML"
    exit 1
fi

# Curated APIs and the migration inventory must be the same set. Do not use
# $TMP_DIR/migrated.txt here: it includes link-script aliases for exported
# symbol coverage, while this gate is specifically about one curated wrapper
# per migration-inventory entry.
python3 "$COVERAGE_CHECK" list-curated \
    --yaml "$CURATED_YAML" | sort -u > "$TMP_DIR/curated.txt"
awk '{print $1}' "$INV" | sort -u > "$TMP_DIR/inventory_names.txt"

MISSING_FROM_CURATED="$(comm -23 "$TMP_DIR/inventory_names.txt" "$TMP_DIR/curated.txt" || true)"
EXTRA_IN_CURATED="$(comm -13 "$TMP_DIR/inventory_names.txt" "$TMP_DIR/curated.txt" || true)"
if [ -n "$MISSING_FROM_CURATED" ] || [ -n "$EXTRA_IN_CURATED" ]; then
    echo "FAIL (curated): HSA curated_apis.yaml must exactly match migration inventory"
    if [ -n "$MISSING_FROM_CURATED" ]; then
        echo "  missing from curated YAML:"
        printf '    %s\n' $MISSING_FROM_CURATED
    fi
    if [ -n "$EXTRA_IN_CURATED" ]; then
        echo "  not in migration inventory:"
        printf '    %s\n' $EXTRA_IN_CURATED
    fi
    exit 1
fi

# Body-content scan: each curated wrapper body must have the sentinel, a
# curated return macro (or the pure-void wrapper's direct shared-exit path),
# and an IN snapshot when it captures IN arguments.
python3 "$COVERAGE_CHECK" curated \
    --src-dir "$HSA_ROOT" \
    --yaml "$CURATED_YAML"

N_MIGRATED=$(wc -l < "$TMP_DIR/migrated.txt")
N_EXPORTED=$(wc -l < "$TMP_DIR/exported.txt")
echo "PASS: all $N_EXPORTED exported HSA symbols migrated ($N_MIGRATED inventory entries); $(wc -l < "$TMP_DIR/curated.txt") curated wrappers verified"
