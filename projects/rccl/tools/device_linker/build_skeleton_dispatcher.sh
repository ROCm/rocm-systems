#!/bin/bash
#
# Build skeleton dispatcher for device linker
#
# This script compiles common.cu with DEVICE_LINKER flags to create a skeleton
# dispatcher containing:
#   - 3 generic kernels (ncclDevKernel_Generic_1/2/4)
#   - Empty function tables (ncclDevFuncTable_1/2/4, 859 entries each)
#   - __shared__ ncclShmemData ncclShmem declaration
#
# Add -DENABLE_COLLTRACE for debug kernels (ncclDevKernelDebug_Generic_1/2/4)
#
# The device linker will later merge this with specialized kernels and populate
# the function tables.
#
# REQUIRED SOURCE CHANGES (already applied to src/device/common.h):
#   1. Line 16: Skip device_table.h when DEVICE_LINKER defined
#      #if !defined(NCCL_SPECIALIZED_KERNEL) && !defined(DEVICE_LINKER)
#   2. After line 18: Add extern declarations for function tables
#      #ifdef DEVICE_LINKER
#      extern __device__ ncclDevFuncPtr_t ncclDevFuncTable_1/2/4[FUNC_COUNT];
#      #endif
#   3. Line 694: Add DEVICE_LINKER to dispatch condition
#      #if defined(USE_INDIRECT_FUNCTION_CALL) || defined(DEVICE_LINKER)
#
# NOTE: If hipify hasn't been run since source changes, you also need to apply
#       these changes to build/release/hipify/src/device/common.h

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$RCCL_ROOT/build/release"
HIPIFY_DIR="$BUILD_DIR/hipify"
OUTPUT_DIR="$SCRIPT_DIR"

# Check that hipified sources exist
if [ ! -d "$HIPIFY_DIR/src/device" ]; then
    echo "Error: Hipified sources not found at $HIPIFY_DIR"
    echo "Run a CMake configure first to generate hipified sources."
    exit 1
fi

# Source file (hipified)
SOURCE="$HIPIFY_DIR/src/device/common.cu.cpp"

# Output
OUTPUT="$OUTPUT_DIR/skeleton_dispatcher.o"

# GPU target
GPU_TARGET="${GPU_TARGET:-gfx942}"

echo "=== Building Skeleton Dispatcher ==="
echo "Source: $SOURCE"
echo "Output: $OUTPUT"
echo "GPU Target: $GPU_TARGET"
echo ""

# Compile flags
HIPCC=/opt/rocm/bin/hipcc
CFLAGS="-c -fPIC -fno-gpu-rdc --offload-arch=$GPU_TARGET"

# Defines - read from environment variables (defaults match typical build)
DEFINES="-DDEVICE_LINKER"
if [ "${ENABLE_FAULT_INJECTION:-1}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_FAULT_INJECTION"
fi
if [ "${ENABLE_WARP_SPEED:-1}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_WARP_SPEED"
fi
if [ "${ENABLE_LL128:-1}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_LL128"
fi
if [ "${ENABLE_COLLTRACE:-1}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_COLLTRACE"
fi
if [ "${ENABLE_PROFILING:-0}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_PROFILING"
fi
echo "Build defines: $DEFINES"

# Include paths (order matters)
INCLUDES=(
    "-I$HIPIFY_DIR/src"
    "-I$HIPIFY_DIR/src/include"
    "-I$HIPIFY_DIR/src/include/nccl_device"
    "-I$HIPIFY_DIR/src/include/plugin"
    "-I$HIPIFY_DIR/src/device"
    "-I$HIPIFY_DIR/gensrc"
    "-I$BUILD_DIR/include"
)

echo "Compiling..."
set -x
$HIPCC $CFLAGS $DEFINES "${INCLUDES[@]}" "$SOURCE" -o "$OUTPUT"
set +x

echo ""
echo "=== Build Complete ==="
echo "Output: $OUTPUT"
echo ""

# Verify the output
echo "=== Verifying Output ==="
echo "File size: $(ls -lh "$OUTPUT" | awk '{print $5}')"
echo ""

# Check for .hip_fatbin section
echo "Sections:"
/opt/rocm/llvm/bin/llvm-objdump -h "$OUTPUT" | grep -E "^\s+[0-9]+" | head -10
echo ""

# Check for kernel symbols
echo "Kernel symbols:"
/opt/rocm/llvm/bin/llvm-nm "$OUTPUT" 2>/dev/null | grep -E "ncclDevKernel|ncclDevFuncTable" | head -20 || echo "(symbols may be in device code)"

echo ""
echo "=== Done ==="
