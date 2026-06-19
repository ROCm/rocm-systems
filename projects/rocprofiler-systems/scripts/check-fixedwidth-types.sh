#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Linter: enforce std-qualified fixed-width integer types (std::uint32_t, not uint32_t)
#         and C++ includes (<cstdint>, not <stdint.h>).
#
# Checks:
#   1. Bare types (uint32_t) → must be std::uint32_t
#   2. C-style include (<stdint.h>) → must be <cstdint>
#   3. Missing <cstdint> when std:: types are used
#
# Usage:
#   ./scripts/check-fixedwidth-types.sh [--fix] [file ...]
#   ./scripts/check-fixedwidth-types.sh                # check source/
#   ./scripts/check-fixedwidth-types.sh --fix          # auto-fix all files
#
# Exit codes:
#   0  all files pass (or --fix applied successfully)
#   1  violations detected
#
# Limitation: this script uses regex, not a real parser. --fix will replace
# matches inside comments and string literals too. Review changes after fixing.

set -euo pipefail

TYPES_RE="uint8_t|uint16_t|uint32_t|uint64_t|int8_t|int16_t|int32_t|int64_t"

FIX=0
files=()

for arg in "$@"; do
    if [[ "$arg" == "--fix" ]]; then
        FIX=1
    else
        files+=("$arg")
    fi
done

if [[ ${#files[@]} -eq 0 ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
    while IFS= read -r -d '' f; do
        files+=("$f")
    done < <(find "${PROJECT_DIR}/source" \
        -type f \( -name '*.cpp' -o -name '*.hpp' \) \
        ! -path '*/external/*' ! -path '*/tpls/*' \
        -print0 2>/dev/null)
fi

has_bare() {
    grep -Pq "(?<!std::)\b(${TYPES_RE})\b" "$1"
}

has_c_include() {
    grep -q '#include *<stdint\.h>' "$1"
}

has_cxx_include() {
    grep -q '#include *<cstdint>' "$1"
}

uses_types() {
    grep -Pq "\b(std::)?(${TYPES_RE})\b" "$1"
}

fix_types() {
    local f="$1"
    perl -pi -e "s/(?<!::)\b(${TYPES_RE})\b/std::\$1/g" "$f"
}

fix_includes() {
    local f="$1"
    if has_c_include "$f"; then
        sed -i '/#include *<stdint\.h>/d' "$f"
    fi
    if uses_types "$f" && ! has_cxx_include "$f"; then
        sed -i '0,/^#include/s//#include <cstdint>\n&/' "$f"
    fi
}

type_violations=()
include_violations=()
fixed_count=0

for f in "${files[@]}"; do
    [[ -f "$f" ]] || continue

    if has_bare "$f"; then
        if [[ $FIX -eq 1 ]]; then
            fix_types "$f"
            fix_includes "$f"
            echo "  fixed: $f"
            fixed_count=$((fixed_count + 1))
        else
            type_violations+=("$f")
        fi
    elif has_c_include "$f"; then
        if [[ $FIX -eq 1 ]]; then
            fix_includes "$f"
            echo "  fixed: $f"
            fixed_count=$((fixed_count + 1))
        else
            include_violations+=("$f:stdint.h")
        fi
    elif uses_types "$f" && ! has_cxx_include "$f"; then
        if [[ $FIX -eq 1 ]]; then
            fix_includes "$f"
            echo "  fixed: $f"
            fixed_count=$((fixed_count + 1))
        else
            include_violations+=("$f:missing-cstdint")
        fi
    fi
done

if [[ $FIX -eq 1 ]]; then
    echo "Fixed ${fixed_count} file(s). Re-run without --fix to verify."
    exit 0
fi

rc=0

if [[ ${#type_violations[@]} -gt 0 ]]; then
    echo "Error: bare fixed-width integer types detected."
    echo "Project convention: use std-qualified types (std::uint32_t, not uint32_t)."
    echo ""
    echo "Files with bare types:"
    for f in "${type_violations[@]}"; do
        echo "  $f"
        grep -Pn "(?<!std::)\b(${TYPES_RE})\b" "$f" | head -5 | sed 's/^/    /'
        echo ""
    done
    rc=1
fi

if [[ ${#include_violations[@]} -gt 0 ]]; then
    echo "Error: C-style or missing fixed-width integer includes detected."
    echo "Project convention: use #include <cstdint>, not #include <stdint.h>."
    echo ""
    for entry in "${include_violations[@]}"; do
        f="${entry%%:*}"
        reason="${entry##*:}"
        if [[ "$reason" == "stdint.h" ]]; then
            echo "  $f — uses <stdint.h>, replace with <cstdint>"
        else
            echo "  $f — uses fixed-width types but missing #include <cstdint>"
        fi
    done
    echo ""
    rc=1
fi

if [[ $rc -ne 0 ]]; then
    echo "Run with --fix to auto-fix: ./scripts/check-fixedwidth-types.sh --fix"
fi

exit $rc
