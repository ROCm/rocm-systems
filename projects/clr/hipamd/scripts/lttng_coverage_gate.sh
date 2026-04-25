#!/usr/bin/env bash
# LTTng symbol-coverage gate.
#
# Diffs the public HIP symbols exported from libamdhip64.so against the
# AST-migration inventory (one or more TSV files written by
# lttng_migrate.py). Fails the build on any exported HIP symbol that wasn't
# migrated AND isn't in the exemption file.
#
# Usage:
#   lttng_coverage_gate.sh <path/to/libamdhip64.so> <inv1.txt> [<inv2.txt> ...]
#
# Exemption file: scripts/lttng_coverage_exemptions.txt (in same dir as
# this script) - listed symbols are skipped. See the file for the format.
#
# Exit codes:
#   0  PASS - all exported HIP symbols are present in the inventory
#   1  FAIL - one or more symbols missing from inventory
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

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

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
    echo "These symbols are exported from $SO but lttng_migrate.py did not"
    echo "rewrite their wrapper bodies. Either:"
    echo "  - extend the migrator (e.g. add a return-type to classify())"
    echo "  - add the wrappers to a source file the migrator scans"
    echo "  - explicitly exempt the symbol with a known-list (only after review)"
    exit 1
fi

NTOTAL="$(wc -l < "$WORK/exported.txt")"
echo "PASS: all $NTOTAL exported HIP symbols migrated"
