#!/usr/bin/env bash
# LTTng coverage gate (HIP).
#
# 1. Symbol coverage: diff the public HIP symbols exported from
#    libamdhip64.so against the migration inventory (one or more TSV
#    files). Fails the build on any exported HIP symbol that wasn't
#    migrated AND isn't in the exemption file.
#
# 2. Curated body-content coverage: every inventory symbol must be in
#    curated_apis.yaml and its wrapper body must contain the per-API sentinel,
#    a per-API _enter emission, a curated return macro (or direct _exit
#    helper for STRUCT/void returns), and
#    safe IN-value capture locals. This is mandatory: HIP has no generic
#    string-discriminated fallback tier.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libamdhip64.so> <inv1.txt> [<inv2.txt> ...]
#
# Exit codes:
#   0  PASS
#   1  FAIL - symbol-coverage or curated body-content gap
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

# Shared Python tooling (project-agnostic; curated_apis.yaml and the
# migration inventory itself remain per-project in $SCRIPT_DIR).
SHARED_SCRIPTS_DIR="$(cd "$SCRIPT_DIR/../../../../shared/lttng/scripts" && pwd)"
COVERAGE_CHECK="$SHARED_SCRIPTS_DIR/lttng_coverage_check.py"

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
    echo "  - add the wrapper to the curated typed-argument migration"
    echo "  - explicitly exempt the symbol via lttng_coverage_exemptions.txt"
    exit 1
fi

# ---------------------------------------------------------------------------
# 2. Full curated body-content coverage
# ---------------------------------------------------------------------------
# Inventory symbol names, deduplicated.
{
    for inv in "$@"; do
        awk '{print $1}' "$inv"
    done
} | sort -u > "$WORK/inventory_names.txt"

CURATED_YAML="$SCRIPT_DIR/curated_apis.yaml"
if [ ! -f "$CURATED_YAML" ]; then
    echo "ERROR: required curated YAML not found: $CURATED_YAML" >&2
    exit 2
fi

# The Python helper enforces an exact inventory/YAML set match before walking
# source bodies for sentinels, curated emissions, and IN capture locals.
set +e
python3 "$COVERAGE_CHECK" curated \
    --src-dir "$SRC_DIR" \
    --yaml "$CURATED_YAML" \
    --names-file "$WORK/inventory_names.txt" \
    > "$WORK/curated.log" 2>&1
CURATED_RC=$?
set -e
cat "$WORK/curated.log"
if [ "$CURATED_RC" -ne 0 ]; then
    echo "FAIL: curated-args coverage gate failed"
    exit 1
fi

NTOTAL="$(wc -l < "$WORK/exported.txt")"
NINVENTORY="$(wc -l < "$WORK/inventory_names.txt")"
echo "PASS: all $NTOTAL exported HIP symbols are inventoried; all $NINVENTORY inventory APIs are curated"
