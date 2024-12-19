#!/bin/bash

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
