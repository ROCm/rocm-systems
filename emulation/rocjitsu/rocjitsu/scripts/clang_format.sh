#!/bin/bash

# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Format all C++ source files using clang-format.
# Run this to ensure all source files follow the project's .clang-format rules.
#
# Usage:
#   ./clang_format.sh [directory ...]
#
# If no directories are given, formats lib/, tests/, and tools/.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

if ! command -v clang-format &>/dev/null; then
  echo "error: clang-format not found in PATH" >&2
  exit 1
fi

# Default directories if none provided.
if [ $# -eq 0 ]; then
  set -- "$PROJECT_ROOT/lib" "$PROJECT_ROOT/tests" "$PROJECT_ROOT/tools"
fi

# Find all .cpp and .h files under the given directories.
files=$(find "$@" -type f \( -name '*.cpp' -o -name '*.h' \))

if [ -z "$files" ]; then
  echo "No C++ files found"
  exit 0
fi

count=$(echo "$files" | wc -l)
clang-format -i -style=file:"$PROJECT_ROOT/.clang-format" $files

echo "Formatted $count files in: $*"
