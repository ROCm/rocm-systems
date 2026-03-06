#!/usr/bin/env bash
if [[ $# -ne 1 ]]; then
    echo "Usage: $0 directory"
    exit 1
fi

TARGET_DIR="$1"
if [[ ! -d "$TARGET_DIR" ]]; then
    echo "Directory $TARGET_DIR does not exist."
    exit 1
fi
if [[ ! -d "control/$TARGET_DIR" ]]; then
    echo "Directory control/$TARGET_DIR does not exist."
    exit 1
fi

cp -r "control/$TARGET_DIR"/* "$TARGET_DIR"/
if [[ $? -ne 0 ]]; then
    echo "Failed to copy files from control/$TARGET_DIR to $TARGET_DIR."
    exit 1
fi

cd "$TARGET_DIR" || exit 1
shopt -s nullglob

# Collect arrays of stats_ and compare_ files
stats_files=(stats_*)
compare_files=(compare_*)

for stats_file in "${stats_files[@]}"; do
    # Remove the "stats_" prefix to form the suffix
    suffix="${stats_file#stats_}"
    compare_file="compare_${suffix}"
    
    if [[ -f "$compare_file" ]]; then
        #diff "$stats_file" "$compare_file"
        python3 ../diff.py "$stats_file" "$compare_file"
    else
        echo "Compare file $compare_file not found for $stats_file"
    fi
done

if [[ ${#stats_files[@]} -ne ${#compare_files[@]} ]]; then
    echo "Mismatch in file counts: ${#stats_files[@]} stats files vs ${#compare_files[@]} compare files."
    exit 1
fi
