#!/usr/bin/env bash
# LTTng coverage gate (HIP).
#
# 1. Symbol coverage: diff the public HIP symbols exported from
#    libamdhip64.so against the migration inventory (one or more TSV
#    files). Fails the build on any exported HIP symbol that wasn't
#    migrated AND isn't in the exemption file.
#
# 2. Body-content coverage (defense-in-depth): for each migrated symbol,
#    locate its function body in the source TUs and require:
#      - >=1 rocm_trace_emit_hip_api_enter call
#      - AND (>=1 ROCM_TRACE_RET_*/ROCR_TRACE_API_RET_* macro
#        OR    >=1 rocm_trace_emit_hip_api_exit_* call)
#    Guards against drift (hand edits, partial reverts).
#    By default body-content failures are reported as WARNINGs only.
#    Set `LTTNG_COVERAGE_STRICT=1` in the environment to make them fatal.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libamdhip64.so> <inv1.txt> [<inv2.txt> ...]
#
# Exit codes:
#   0  PASS
#   1  FAIL - symbol-coverage gap, or body-content gap with STRICT=1
#   2  USAGE error
set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: $0 <path/to/libamdhip64.so> <inventory1.txt> [<inventory2.txt> ...]" >&2
    exit 2
fi

SO="$1"; shift
if [ ! -f "$SO" ]; then
    echo "ERROR: $SO not found" >&2
    exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
EXEMPT_FILE="$SCRIPT_DIR/lttng_coverage_exemptions.txt"

# Source-tree root that contains the wrapper TUs. The HIP wrappers live in
# projects/clr/hipamd/src; the script itself is in projects/clr/hipamd/scripts.
SRC_DIR="$(cd "$SCRIPT_DIR/.." && pwd)/src"

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# ---------------------------------------------------------------------------
# 1. Symbol-coverage gate
# ---------------------------------------------------------------------------

# Exported text symbols (T or t) starting with hip or __hip. Strip
# version-script suffix (`name@@version` -> `name`).
nm -D --defined-only "$SO" \
    | awk '$2 ~ /^[Tt]$/ && $3 ~ /^(hip|__hip)/ { sub(/@@.*/, "", $3); print $3 }' \
    | sort -u > "$WORK/exported.txt"

# Migrated symbols (column 1 of each inventory file) plus exemptions.
{
    for inv in "$@"; do
        if [ ! -f "$inv" ]; then
            echo "ERROR: inventory $inv not found" >&2
            exit 2
        fi
        awk '{print $1}' "$inv"
    done
    if [ -f "$EXEMPT_FILE" ]; then
        # Strip comments and blanks.
        sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$EXEMPT_FILE"
    fi
} | sort -u > "$WORK/migrated.txt"

# Symbols in .so but NOT in any inventory.
MISSING="$(comm -23 "$WORK/exported.txt" "$WORK/migrated.txt" || true)"
if [ -n "$MISSING" ]; then
    NMISS="$(printf '%s\n' "$MISSING" | wc -l)"
    echo "FAIL: $NMISS exported HIP symbols are NOT in the LTTng migration inventory:"
    printf '  %s\n' $MISSING
    echo ""
    echo "These symbols are exported from $SO but their wrapper bodies"
    echo "are not in the LTTng migration inventory. Either:"
    echo "  - migrate the wrapper (insert rocm_trace_emit_hip_api_enter/exit calls)"
    echo "  - explicitly exempt the symbol via lttng_coverage_exemptions.txt"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Body-content coverage (defense-in-depth)
# ---------------------------------------------------------------------------
# Inventory symbol names, deduplicated.
{
    for inv in "$@"; do
        awk '{print $1}' "$inv"
    done
} | sort -u > "$WORK/inventory_names.txt"

STRICT="${LTTNG_COVERAGE_STRICT:-0}"
N_BODY_FAIL=0

# Body-content gate driven by lttng_coverage_check.py (subcommand `body`).
# The Python helper walks SRC_DIR for .cpp wrappers and verifies each
# inventoried symbol's body contains both an ENTER call and a balancing
# EXIT macro/helper. See lttng_coverage_check.py for full semantics.
set +e
python3 "$SCRIPT_DIR/lttng_coverage_check.py" body \
    --src-dir "$SRC_DIR" \
    --names-file "$WORK/inventory_names.txt" \
    > "$WORK/body.log" 2>&1
BODY_RC=$?
set -e
cat "$WORK/body.log"

if [ "$BODY_RC" -ne 0 ]; then
    if [ "$STRICT" = "1" ]; then
        echo "FAIL: body-content coverage gate failed (LTTNG_COVERAGE_STRICT=1)"
        exit 1
    else
        echo "WARN: body-content coverage gate found gaps (set LTTNG_COVERAGE_STRICT=1 to fail the build)"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Curated-args coverage gate
# ---------------------------------------------------------------------------
# Skipped silently when curated_apis.yaml is absent (allows gradual
# rollout; the gate becomes mandatory once any API is curated).

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ -f "$CURATED_YAML" ]; then
    # Curated APIs (one per line). Drives the inventory-membership check
    # below; the actual body-content scan lives in lttng_coverage_check.py.
    python3 "$SCRIPT_DIR/lttng_coverage_check.py" list-curated \
        --yaml "$CURATED_YAML" | sort -u > "$WORK/curated.txt"

    # All inventoried wrappers (already in $WORK/migrated.txt from gate 1).
    MISSING_FROM_INV="$(comm -23 "$WORK/curated.txt" "$WORK/migrated.txt" || true)"
    if [ -n "$MISSING_FROM_INV" ]; then
        echo "FAIL (curated): APIs in curated_apis.yaml are missing from migration inventory:"
        printf '  %s\n' $MISSING_FROM_INV
        exit 1
    fi

    # Body-content scan: each curated wrapper body must have the sentinel,
    # a _CURATED macro invocation, and (for APIs with any IN/INOUT arg) at
    # least one __rocm_in_ local. See lttng_coverage_check.py:gate_curated
    # for the implementation.
    set +e
    python3 "$SCRIPT_DIR/lttng_coverage_check.py" curated \
        --src-dir "$SRC_DIR" \
        --yaml "$CURATED_YAML" \
        > "$WORK/curated.log" 2>&1
    CURATED_RC=$?
    set -e
    cat "$WORK/curated.log"
    if [ "$CURATED_RC" -ne 0 ]; then
        echo "FAIL: curated-args coverage gate failed"
        exit 1
    fi
fi

NTOTAL="$(wc -l < "$WORK/exported.txt")"
echo "PASS: all $NTOTAL exported HIP symbols migrated"
