#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# CI Script
# ./files-changed.sh BASE_REF HEAD_REF PATTERN
# Prints 0 if no matching files have changed
# Prints 1 if at least 1 matched file has changed.

set -eo pipefail

BASE_REF=$1
HEAD_REF=$2
CHECK_PATTERN=$3
FILES_CHANGED=0

changed_files=()
if ! diff_output="$(git diff --name-only "${BASE_REF}" "${HEAD_REF}")"; then
    echo "${diff_output}"
    exit 1
fi
mapfile -t changed_files <<< "${diff_output}"

for changed_file in "${changed_files[@]}"; do
    # This is intended behaviour to check for globbing.
    # shellcheck disable=SC2053
    if [[ "${changed_file}" == ${CHECK_PATTERN} ]]; then
        FILES_CHANGED=1
        break
    fi
done

printf '%s' ${FILES_CHANGED}
exit 0
