#!/usr/bin/env bash

set -e
set -u
set -o pipefail

if ! command -v gersemi >/dev/null 2>&1; then
    echo "$(tput setaf 1)gersemi is not installed. Please install it to format CMake files.$(tput sgr0)"
    echo -e "$(tput setaf 6)try this:\n    python3 -m pip install gersemi$(tput sgr0)"
    exit 1
fi

FILES=$(find . -type f \( -name "CMakeLists.txt" -o -name "*.cmake" \) \
    -not -name "*.in" \
    -not -path "*/\.*" \
    -not -path "*/build/*")

failed_files=()

# Check if files are formatted correctly
for file in $FILES; do
    echo "Checking $file..."
    if ! gersemi --check "$file"; then
        failed_files+=("$file")
        echo "::error file=$file::File needs formatting"
    fi
done

if [ ${#failed_files[@]} -ne 0 ]; then
    gersemi -i "${failed_files[@]}"
fi
