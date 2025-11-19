#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CUID_TOOL="$SCRIPT_DIR/../build/cli/amdcuid_tool"

if [ ! -f "$CUID_TOOL" ]; then
    echo "Error: CUID tool not found. Build the project first."
    exit 1
fi

TEST_DIR=$(mktemp -d)
trap "rm -rf $TEST_DIR" EXIT

KEY_FILE="$TEST_DIR/hmac_key.bin"
CUID_FILE="$TEST_DIR/cuid"

dd if=/dev/urandom of="$KEY_FILE" bs=32 count=1 2>/dev/null
chmod 600 "$KEY_FILE"

if ! "$CUID_TOOL" --generate-cuid "$KEY_FILE" --output-file "$CUID_FILE" 2>&1 | grep -q "Successfully generated"; then
    echo "FAIL: CUID generation failed"
    exit 1
fi

if [ ! -f "$CUID_FILE" ]; then
    echo "FAIL: CUID file not created"
    exit 1
fi

if ! grep -q "\[GPU:" "$CUID_FILE"; then
    echo "FAIL: Invalid file format"
    exit 1
fi

if ! "$CUID_TOOL" --list-file "$CUID_FILE" > /dev/null; then
    echo "FAIL: Cannot parse CUID file"
    exit 1
fi

echo "PASS: All tests passed"
