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

# Output file name
OUTPUT_FILE="merged.pftrace"
# Legacy output file name, excluded below so a stale merged.proto from a
# pre-rename run is not folded back in as if it were an unmerged per-process
# trace.
LEGACY_OUTPUT_FILE="merged.proto"

# Collect the per-process traces. '.proto' is still accepted so that output
# directories produced by older versions, or by an explicitly configured
# ROCPROFSYS_PERFETTO_FILE, still merge. Previously merged output (current or
# legacy name) is excluded so that re-running the merge does not fold its own
# result back in.
shopt -s nullglob
TRACE_FILES=()
for file in "$FOLDER_PATH"/*.pftrace "$FOLDER_PATH"/*.proto; do
  base=$(basename "$file")
  if [ "$base" != "$OUTPUT_FILE" ] && [ "$base" != "$LEGACY_OUTPUT_FILE" ]; then
    TRACE_FILES+=("$file")
  fi
done

# Check if there is more than one trace file
if [ ${#TRACE_FILES[@]} -le 1 ]; then
  exit 0
fi

echo "Merging multiprocess files ..."
# Check if all trace files have been fully written or wait
TIMEOUT=60  # Timeout in seconds
for file in "${TRACE_FILES[@]}"; do
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

# Merge all trace files into one file
cat "${TRACE_FILES[@]}" > "$FOLDER_PATH"/"$OUTPUT_FILE"

echo "All multiprocess trace files in '$FOLDER_PATH' have been merged into '$OUTPUT_FILE'."
