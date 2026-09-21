#!/bin/bash

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <folder>"
    exit 1
fi

FOLDER="$1"

if [ ! -d "$FOLDER" ]; then
    echo "Error: '$FOLDER' is not a directory"
    exit 1
fi

for full_path in "$FOLDER"/*; do
    [ -f "$full_path" ] || continue

    file_name="$(basename "$full_path")"

    echo "=== Profiling: $file_name ==="
    ./src/rocprof-compute profile --roof-only \
        --output-directory "/work-space/output/bmm/$file_name" \
        --overwrite \
        -- python "$full_path"

    echo "=== Analyzing: $file_name ==="
    ./src/rocprof-compute analyze -p "/work-space/output/bmm/$file_name"
done
