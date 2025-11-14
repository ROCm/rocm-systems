#!/bin/bash

#  Copyright © Advanced Micro Devices, Inc., or its affiliates.
# 
#  SPDX-License-Identifier: NCSA
# 

################################################################################
# GPU Core Dump Pipe Handler Test Script
#
# This script can be used to test the GPU core dump pipe pattern functionality.
# It reads a GPU core dump from stdin, validates it, and saves it to a file.
#
# Usage:
#   1. Set as core pattern: 
#      sudo sysctl -w kernel.core_pattern="|/path/to/test_gpu_core_pipe_handler.sh %p %e %t"
#   2. Trigger GPU fault
#   3. Check output in /tmp/gpu_core_dumps/
#
# Arguments (from core_pattern):
#   $1 = PID
#   $2 = Executable name
#   $3 = Timestamp
################################################################################

# Configuration
OUTPUT_DIR="/tmp/gpu_core_dumps"
LOG_FILE="${OUTPUT_DIR}/handler.log"

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Get arguments
PID="${1:-unknown}"
EXE="${2:-unknown}"
TIMESTAMP="${3:-$(date +%s)}"

# Generate output filename
OUTPUT_FILE="${OUTPUT_DIR}/gpu_core.${EXE}.${PID}.${TIMESTAMP}"

# Log start
echo "[$(date)] GPU Core Dump Handler Started" >> "${LOG_FILE}"
echo "  PID: ${PID}" >> "${LOG_FILE}"
echo "  Executable: ${EXE}" >> "${LOG_FILE}"
echo "  Timestamp: ${TIMESTAMP}" >> "${LOG_FILE}"
echo "  Output: ${OUTPUT_FILE}" >> "${LOG_FILE}"

# Read from stdin and save to file
cat > "${OUTPUT_FILE}"

# Check if we received data
if [ ! -s "${OUTPUT_FILE}" ]; then
    echo "  ERROR: No data received from stdin" >> "${LOG_FILE}"
    exit 1
fi

# Get file size
FILE_SIZE=$(stat -f%z "${OUTPUT_FILE}" 2>/dev/null || stat -c%s "${OUTPUT_FILE}" 2>/dev/null)
echo "  File size: ${FILE_SIZE} bytes" >> "${LOG_FILE}"

# Validate ELF header
if command -v readelf &> /dev/null; then
    # Check if it's a valid ELF file
    if readelf -h "${OUTPUT_FILE}" &> /dev/null; then
        MACHINE=$(readelf -h "${OUTPUT_FILE}" | grep "Machine:" | awk '{print $2}')
        TYPE=$(readelf -h "${OUTPUT_FILE}" | grep "Type:" | awk '{print $2}')
        
        echo "  ELF Type: ${TYPE}" >> "${LOG_FILE}"
        echo "  Machine: ${MACHINE}" >> "${LOG_FILE}"
        
        # Check if it's an AMDGPU core dump
        if echo "${MACHINE}" | grep -q "AMDGPU"; then
            echo "  ✓ Valid AMDGPU core dump" >> "${LOG_FILE}"
        else
            echo "  WARNING: Not an AMDGPU core dump (Machine: ${MACHINE})" >> "${LOG_FILE}"
        fi
        
        if [ "${TYPE}" = "CORE" ]; then
            echo "  ✓ Valid core dump type" >> "${LOG_FILE}"
        else
            echo "  WARNING: Not a core dump (Type: ${TYPE})" >> "${LOG_FILE}"
        fi
    else
        echo "  ERROR: Invalid ELF file" >> "${LOG_FILE}"
        exit 1
    fi
else
    # Fallback: check ELF magic bytes
    MAGIC=$(xxd -l 4 -p "${OUTPUT_FILE}")
    if [ "${MAGIC}" = "7f454c46" ]; then
        echo "  ✓ Valid ELF magic number" >> "${LOG_FILE}"
    else
        echo "  ERROR: Invalid ELF magic (got: ${MAGIC})" >> "${LOG_FILE}"
        exit 1
    fi
fi

echo "  ✓ GPU core dump saved successfully" >> "${LOG_FILE}"
echo "[$(date)] Handler Completed Successfully" >> "${LOG_FILE}"
echo "" >> "${LOG_FILE}"

# Print summary to stderr so it appears in logs
echo "GPU Core Dump Handler: Saved ${FILE_SIZE} bytes to ${OUTPUT_FILE}" >&2

exit 0
