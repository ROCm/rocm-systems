#!/bin/bash

##############################################################################
# generate_hmac_key.sh
# 
# This script generates a 256-bit (32 bytes) random key for HMAC-SHA256
# The key should be kept secure and consistent across your organization
##############################################################################

KEY_FILE="${1:-hmac_key.bin}"

echo "Generating 256-bit HMAC key..."

# Generate 32 bytes (256 bits) of random data
dd if=/dev/urandom of="$KEY_FILE" bs=32 count=1 2>/dev/null

if [ $? -eq 0 ]; then
    chmod 600 "$KEY_FILE"
    echo "✓ HMAC key generated: $KEY_FILE"
    echo "✓ File permissions set to 600 (owner read/write only)"
    echo ""
    echo "Key fingerprint (SHA256):"
    sha256sum "$KEY_FILE"
    echo ""
    echo "⚠ IMPORTANT: Keep this key secure and backed up!"
    echo "⚠ The same key must be used across all systems in your cluster"
    echo "⚠ Loss of this key means you cannot regenerate matching secondary CUIDs"
else
    echo "✗ Failed to generate HMAC key"
    exit 1
fi
