#!/usr/bin/env bash

# SPDX-License-Identifier: MIT
# Copyright (C) Advanced Micro Devices. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
# the Software, and to permit persons to whom the Software is furnished to do so,
# subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
# FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
# IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
# CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 

set -e
set -u
set -o pipefail

FILES=$(find . -type f \( -name "CMakeLists.txt" -o -name "*.cmake" -o -name "*.cmake.in" \) \
    -not -path "*/esmi_ib_library/*" \
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
