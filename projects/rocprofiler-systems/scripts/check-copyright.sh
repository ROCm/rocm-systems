#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Pre-commit hook: verify staged files contain an SPDX copyright header.
# Expected header (first 5 lines):
#   // Copyright (c) Advanced Micro Devices, Inc.
#   // SPDX-License-Identifier: MIT
# or the # equivalent for Python/CMake files.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FIX_SCRIPT="$SCRIPT_DIR/fix_license_headers.py"

expected_copyright=("Copyright (c)" ".COPYRIGHT")
files_with_missing_copyright=()
files=("$@")

if [[ "$ALLOW_MISSING_COPYRIGHT" == "1" ]]; then
    exit 0
fi

for file in "${files[@]}"; do
    if [[ -f "$file" ]]; then
        found=0
        for pattern in "${expected_copyright[@]}"; do
            if head -5 "$file" | grep -Fq "$pattern"; then
                found=1
                break
            fi
        done
        if [[ $found -eq 0 ]]; then
            files_with_missing_copyright+=("$file")
        fi
    fi
done

if [ ${#files_with_missing_copyright[@]} -ne 0 ]; then
    echo "Files missing SPDX copyright header:"
    for file in "${files_with_missing_copyright[@]}"; do
        echo "  $file"
    done

    if [[ -f "$FIX_SCRIPT" ]] && command -v python3 &>/dev/null; then
        echo ""
        echo "Applying fix automatically..."
        python3 "$FIX_SCRIPT" "${files_with_missing_copyright[@]}" 2>&1
        echo ""
        echo "Copyright headers added. Please review and re-stage the changes."
    else
        echo ""
        echo "Run 'python3 scripts/fix_license_headers.py' to fix manually."
        if ! command -v python3 &>/dev/null; then
            echo "Note: python3 is required for automatic fix but was not found."
        fi
    fi

    echo "To skip this check, set ALLOW_MISSING_COPYRIGHT=1"
    exit 1
fi

exit 0
