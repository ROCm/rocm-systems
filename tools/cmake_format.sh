#!/usr/bin/env bash

set -e
set -u
set -o pipefail

FILES=$(find . -type f \( -name "CMakeLists.txt" -o -name "*.cmake" -o -name "*.cmake.in" \) \
    -not -path "*/\.*" \
    -not -path "*/build/*")

failed_files=()

# Check if files are formatted correctly
for file in $FILES; do
    echo "Checking $file..."
    if ! cmake-format --check "$file"; then
        failed_files+=("$file")
        echo "::error file=$file::File needs formatting"
    fi
done

if [ ${#failed_files[@]} -ne 0 ]; then
    cmake-format -i "${failed_files[@]}"
fi
