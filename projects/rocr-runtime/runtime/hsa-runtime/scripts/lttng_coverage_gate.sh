#!/usr/bin/env bash
# LTTng coverage gate (HSA).
#
# 1. Symbol coverage: diff the public HSA symbols exported from
#    libhsa-runtime64.so against the LTTng migration inventory.
#    Fails the build on any miss.
#
# 2. Body-content coverage (defense-in-depth): for each migrated HSA
#    wrapper, confirm its function body contains:
#      - >=1 rocm_trace_emit_hsa_api_enter call
#      - AND (>=1 ROCR_TRACE_API_RET_* macro
#        OR    >=1 rocm_trace_emit_hsa_api_exit_* call)
#    Guards against drift (hand edits, partial reverts). WARN by default;
#    set LTTNG_COVERAGE_STRICT=1 to make body-content gaps fatal.
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
# 2. Body-content gate (defense-in-depth)
# ---------------------------------------------------------------------------
STRICT="${LTTNG_COVERAGE_STRICT:-0}"

# Inventory symbol names (first column of the TSV), deduplicated.
awk '{print $1}' "$INV" | sort -u > "$TMP_DIR/inventory_names.txt"

# Body-content gate driven by lttng_coverage_check.py (subcommand `body`),
# shared with HIP's coverage gate. Walks $HSA_ROOT for .cpp wrappers (every
# migrated HSA wrapper lives in hsa_table_interface.cpp) and verifies each
# inventoried symbol's body contains both an ENTER call and a balancing
# EXIT macro/helper. See lttng_coverage_check.py for full semantics.
set +e
python3 "$COVERAGE_CHECK" body \
    --src-dir "$HSA_ROOT" \
    --names-file "$TMP_DIR/inventory_names.txt" \
    > "$TMP_DIR/body.log" 2>&1
BODY_RC=$?
set -e
cat "$TMP_DIR/body.log"

if [ "$BODY_RC" -ne 0 ]; then
    if [ "$STRICT" = "1" ]; then
        echo "FAIL: body-content coverage gate failed (LTTNG_COVERAGE_STRICT=1)"
        exit 1
    else
        echo "WARN: body-content coverage gate found gaps (set LTTNG_COVERAGE_STRICT=1 to fail the build)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. HSA curated-args coverage gate
# ---------------------------------------------------------------------------
# Skipped silently when curated_apis.yaml is absent (allows gradual
# rollout; the gate becomes mandatory once any API is curated).

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    # Curated APIs (one per line). Drives the inventory-membership check
    # below; the actual body-content scan lives in lttng_coverage_check.py.
    python3 "$COVERAGE_CHECK" list-curated \
        --yaml "$CURATED_YAML" | sort -u > "$TMP_DIR/curated.txt"

    # All inventoried wrappers (already in $TMP_DIR/migrated.txt from gate 1).
    MISSING_FROM_INV="$(comm -23 "$TMP_DIR/curated.txt" "$TMP_DIR/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in HSA curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    # Body-content scan: each curated wrapper body must have the sentinel,
    # a _CURATED_HSA macro invocation, and (for APIs with
    # any IN/INOUT arg) at least one __rocm_in_ local. The shared
    # lttng_coverage_check.py:gate_curated regex matches both HIP
    # (_CURATED) and HSA (_CURATED_HSA) variants, so the same script
    # serves both providers.
    set +e
    python3 "$COVERAGE_CHECK" curated \
        --src-dir "$HSA_ROOT" \
        --yaml "$CURATED_YAML" \
        > "$TMP_DIR/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$TMP_DIR/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: HSA curated-args coverage gate failed"
        exit 1
    fi
fi

N_MIGRATED=$(wc -l < "$TMP_DIR/migrated.txt")
N_EXPORTED=$(wc -l < "$TMP_DIR/exported.txt")
echo "PASS: all $N_EXPORTED exported HSA symbols migrated ($N_MIGRATED inventory entries)"
