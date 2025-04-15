#!/bin/bash

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

# Check if the folder path is provided
if [ -z "$1" ]; then
  echo "Usage: $0 <folder_path>"
  exit 1
fi

# Assign the folder path to a variable
FOLDER_PATH=$1

# Check if the folder exists
if [ ! -d "$FOLDER_PATH" ]; then
  echo "Error: Folder '$FOLDER_PATH' does not exist."
  exit 1
fi

# Check if there are more than one .proto files
PROTO_FILES=("$FOLDER_PATH"/*.proto)
if [ ${#PROTO_FILES[@]} -le 1 ]; then
  exit 0
fi

echo "Merging multiprocess files ..."
# Check if all .proto files have been fully written or wait
TIMEOUT=60  # Timeout in seconds
for file in "${PROTO_FILES[@]}"; do
  SECONDS=0
  while lsof "$file" > /dev/null 2>&1; do
    if [ $SECONDS -ge $TIMEOUT ]; then
      echo "Timeout reached while waiting for $file to be released."
      break
    fi
    echo "Waiting for $file to be released..."
    sleep 1
  done
done

# Output file name
OUTPUT_FILE="merged.proto"

# Merge all .proto files into one file
cat "$FOLDER_PATH"/*.proto > "$FOLDER_PATH"/"$OUTPUT_FILE"

echo "All multiprocess .proto files in '$FOLDER_PATH' have been merged into '$OUTPUT_FILE'."
