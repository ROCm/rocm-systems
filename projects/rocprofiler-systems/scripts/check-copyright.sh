#!/usr/bin/env bash

# MIT License
#
# Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

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
            if grep -Fq "$pattern" "$file"; then
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
    if [[ "$ADD_COPYRIGHT" == "1" ]]; then
        for file in "${files_with_missing_copyright[@]}"; do
            # Determine the comment style based on the file extension
            if [[ "$file" == *.c || "$file" == *.cpp || "$file" == *.h || "$file" == *.hpp ]]; then
                comS="//"
            else
                comS="#"
            fi

            # Read LICENSE file and prepend comment prefix to each line
            copyright_notice=""
            while IFS= read -r line; do
                if [[ -n "$line" ]]; then
                    copyright_notice+="$comS $line"$'\n'
                else
                    copyright_notice+="$comS"$'\n'
                fi
            done < LICENSE

            # Add the notice to the beginning of the file
            temp_file=$(mktemp)
            {
                echo -e "$copyright_notice"
                cat "$file"
            } > "$temp_file"
            mv "$temp_file" "$file"
        done
        echo "Copyright notices added."
        exit 1
    fi
    echo "The following files are missing a valid copyright notice:"
    echo ""
    for file in "${files_with_missing_copyright[@]}"; do
        echo "$file"
    done
    echo ""
    echo "It may be the case that the copyright is not required by some files."
    echo "To override this check, set the environment variable ALLOW_MISSING_COPYRIGHT=1"
    echo "To add the copyright to all files listed above, set the environment variable ADD_COPYRIGHT=1"
    exit 1
fi

exit 0
