#!/usr/bin/env bash
# Copyright (c) Advanced Micro Devices, Inc. All rights reserved.
#
# SPDX-License-Identifier: MIT

# Recursively format all C & C++ sources and header files

COMMAND="clang-format"

DIRS=(
    "examples"
    "src"
    "test"
    "tools"
)

if [ $# -eq 1 ]; then
    COMMAND="$COMMAND-$1"
fi

echo ""
echo "format-source <version>"
echo ""
echo "Format the C and C++ source using clang-format. The <version>"
echo "parameter is optional and can be used to force a specific"
echo "installed version of clang-format to be used."
echo ""

find "${DIRS[@]}" -iname "*.h" -or -iname "*.c" -or -iname "*.cpp" \
    | xargs -P0 -n1 "${COMMAND}" -style=file -i -fallback-style=none

exit 0
