#!/bin/bash
#
# Build dispatcher object using device linker (Option B)
#
# Pipeline:
# 1. Compile dispatcher device code (clang -cc1 amdgcn) → device .o
# 2. Link dispatcher device code (lld) → device ELF
# 3. Device linker merges dispatcher + specialized kernels → merged device ELF
# 4. Bundle merged device ELF (clang-offload-bundler) → .hipfb
# 5. Host compilation with -fcuda-include-gpubinary → final .o
#
# Usage: build_with_device_linker.sh [BUILD_DIR] [GPU_TARGET]
#   BUILD_DIR: CMake build directory (default: ../../build/release)
#   GPU_TARGET: GPU architecture (default: gfx942)
#

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Accept BUILD_DIR as first argument or use default
BUILD_DIR="${1:-$RCCL_ROOT/build/release}"
HIPIFY_DIR="$BUILD_DIR/hipify"

# Accept GPU_TARGET as second argument or use default
GPU_TARGET_FULL="${2:-${GPU_TARGET:-gfx942}}"
# Strip feature flags for -target-cpu (e.g., gfx942:xnack+:sramecc+ -> gfx942)
GPU_TARGET=$(echo "$GPU_TARGET_FULL" | sed 's/:.*$//')
# Keep full target for bundler (e.g., gfx942:sramecc+:xnack+)
# Extract features after the first colon (if any)
if [[ "$GPU_TARGET_FULL" == *:* ]]; then
    GPU_TARGET_FEATURES=$(echo "$GPU_TARGET_FULL" | sed 's/^[^:]*://' | tr ':' '\n' | sort | tr '\n' ':' | sed 's/:$//')
else
    GPU_TARGET_FEATURES=""
fi
if [ -z "$GPU_TARGET_FEATURES" ]; then
    GPU_TARGET_FEATURES="sramecc+:xnack-"
fi
BUNDLE_TARGET="hipv4-amdgcn-amd-amdhsa--${GPU_TARGET}:${GPU_TARGET_FEATURES}"

# Output directory - use build directory
OUTPUT_DIR="$BUILD_DIR/device_linker_output"
mkdir -p "$OUTPUT_DIR"

# Tools - derive paths from amdclang++
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
AMDCLANG="$ROCM_PATH/bin/amdclang++"
CLANG_RESOURCE_DIR=$($AMDCLANG -print-resource-dir)
LLVM_PATH=$(dirname $(dirname $(dirname "$CLANG_RESOURCE_DIR")))
CLANG="$LLVM_PATH/bin/clang"
LLD="$LLVM_PATH/bin/lld"
BUNDLER="$LLVM_PATH/bin/clang-offload-bundler"
# Bitcode may be in a different clang version dir - find it
if [ -d "$CLANG_RESOURCE_DIR/lib/amdgcn/bitcode" ]; then
    BITCODE_DIR="$CLANG_RESOURCE_DIR/lib/amdgcn/bitcode"
else
    # Search for ocml.bc to find the correct bitcode directory
    BITCODE_DIR=$(dirname $(find "$LLVM_PATH/lib/clang" -name "ocml.bc" 2>/dev/null | head -1))
fi
DEVICE_LINKER="$SCRIPT_DIR/device_linker"

# BUNDLE_TARGET is set above based on GPU_TARGET_FULL

# Source and output files
# Use combined dispatcher that includes both common.cu and onerank.cu
# This ensures __clang_gpu_used_external is generated correctly
SOURCE="$SCRIPT_DIR/dispatcher_combined.hip"
DEVICE_OBJ="$OUTPUT_DIR/dispatcher_device.o"
DEVICE_ELF="$OUTPUT_DIR/dispatcher_device.elf"
MERGED_ELF="$OUTPUT_DIR/merged_device.elf"
FATBIN="$OUTPUT_DIR/merged.hipfb"
FINAL_OBJ="$OUTPUT_DIR/dispatcher_final.o"

# Specialized kernel directory (from CMake build)
SPECIALIZED_OBJ_DIR="$BUILD_DIR/specialized_objs"

# Host table for funcId mapping
HOST_TABLE="$HIPIFY_DIR/gensrc/host_table.cpp"

# Check prerequisites
if [ ! -f "$SOURCE" ]; then
    echo "Error: Source not found: $SOURCE"
    exit 1
fi

# Check that hipified sources exist (dispatcher_combined.hip includes them)
if [ ! -f "$HIPIFY_DIR/src/device/common.cu.cpp" ]; then
    echo "Error: Hipified common.cu not found: $HIPIFY_DIR/src/device/common.cu.cpp"
    echo "Run CMake configure first to generate hipified sources."
    exit 1
fi

if [ ! -f "$HIPIFY_DIR/src/device/onerank.cu.cpp" ]; then
    echo "Error: Hipified onerank.cu not found: $HIPIFY_DIR/src/device/onerank.cu.cpp"
    echo "Run CMake configure first to generate hipified sources."
    exit 1
fi

if [ ! -f "$DEVICE_LINKER" ]; then
    echo "Error: Device linker not found: $DEVICE_LINKER"
    exit 1
fi

# Include paths
INCLUDES=(
    "-I$HIPIFY_DIR/src"
    "-I$HIPIFY_DIR/src/include"
    "-I$HIPIFY_DIR/src/include/nccl_device"
    "-I$HIPIFY_DIR/src/include/plugin"
    "-I$HIPIFY_DIR/src/device"
    "-I$HIPIFY_DIR/gensrc"
    "-I$BUILD_DIR/include"
)

# Defines
# IMPORTANT: These must match host compilation (CMakeLists.txt) to ensure structure layout agreement
# All of these affect struct layouts or code paths and MUST be consistent:
#   - DEVICE_LINKER: Uses function tables instead of direct calls
#   - ENABLE_FAULT_INJECTION: Adds faults field to ncclShmemData
#   - ENABLE_WARP_SPEED: Affects ncclShmemData layout (warpComm, warpChannel fields) and kernel args size
#   - ENABLE_LL128: Affects protocol selection and scratch sizes
DEFINES="-DDEVICE_LINKER -DENABLE_FAULT_INJECTION -DENABLE_WARP_SPEED -DENABLE_LL128"

# Debug flag for peer pointer tracking (set DEBUG_PEER_POINTERS=1 to enable)
if [ "${DEBUG_PEER_POINTERS:-0}" = "1" ]; then
    DEFINES="$DEFINES -DDEBUG_PEER_POINTERS"
    echo "DEBUG_PEER_POINTERS enabled - will print peer pointer info from device code"
fi

# CUID (compilation unit ID) - needs to match between device and host compilation
CUID="devicelinker$(echo -n "$SOURCE" | md5sum | cut -c1-16)"

echo "=== Building with Device Linker (Option B) ==="
echo "Source: $SOURCE"
echo "GPU Target: $GPU_TARGET"
echo "CUID: $CUID"
echo ""

# System include paths (from hipcc -### output)
SYS_INCLUDES=(
    "-internal-isystem" "$CLANG_RESOURCE_DIR/include/cuda_wrappers"
    "-idirafter" "$ROCM_PATH/include"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward"
    "-internal-isystem" "$CLANG_RESOURCE_DIR/include"
    "-internal-isystem" "/usr/local/include"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include"
    "-internal-externc-isystem" "/usr/include/x86_64-linux-gnu"
    "-internal-externc-isystem" "/include"
    "-internal-externc-isystem" "/usr/include"
)

# Step 1: Compile dispatcher device code
echo "Step 1: Compiling device code..."
# Derive ISA version from GPU target (e.g., gfx950 -> 950)
ISA_VERSION=$(echo "$GPU_TARGET" | sed 's/gfx//')
$CLANG -cc1 \
    -triple amdgcn-amd-amdhsa \
    -aux-triple x86_64-unknown-linux-gnu \
    -O3 \
    -emit-obj \
    -debug-info-kind=line-tables-only \
    -fcuda-is-device \
    -fno-threadsafe-statics \
    -mllvm -amdgpu-internalize-symbols \
    -fvisibility=hidden \
    -fapply-global-visibility-to-externs \
    -aux-target-cpu x86-64 \
    -mlink-builtin-bitcode "$BITCODE_DIR/ocml.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/ockl.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_daz_opt_off.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_unsafe_math_off.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_finite_only_off.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_correctly_rounded_sqrt_on.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_wavefrontsize64_on.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_isa_version_${ISA_VERSION}.bc" \
    -mlink-builtin-bitcode "$BITCODE_DIR/oclc_abi_version_600.bc" \
    -target-cpu $GPU_TARGET \
    -resource-dir "$CLANG_RESOURCE_DIR" \
    "${SYS_INCLUDES[@]}" \
    -include __clang_hip_runtime_wrapper.h \
    $DEFINES \
    "${INCLUDES[@]}" \
    -fhip-new-launch-api \
    -fgnuc-version=4.2.1 \
    -fcxx-exceptions -fexceptions \
    -cuid=$CUID \
    -fcuda-allow-variadic-functions \
    -o "$DEVICE_OBJ" \
    -x hip "$SOURCE"

echo "  Output: $DEVICE_OBJ ($(ls -lh "$DEVICE_OBJ" | awk '{print $5}'))"

# Step 2: Link dispatcher device code
echo "Step 2: Linking device code..."
$LLD -flavor gnu \
    -m elf64_amdgpu \
    --no-undefined \
    -shared \
    -plugin-opt=-amdgpu-internalize-symbols \
    --lto-partitions=8 \
    -plugin-opt=mcpu=$GPU_TARGET \
    -plugin-opt=O3 \
    --lto-CGO3 \
    --whole-archive \
    -o "$DEVICE_ELF" \
    "$DEVICE_OBJ" \
    --no-whole-archive

echo "  Output: $DEVICE_ELF ($(ls -lh "$DEVICE_ELF" | awk '{print $5}'))"

# Step 3: Device linker merges dispatcher + specialized kernels + onerank
echo "Step 3: Running device linker..."
DEVICE_LINKER_ARGS=(
    -o "$MERGED_ELF"
    --dispatcher "$DEVICE_ELF"
    --host-table "$HOST_TABLE"
    --target "${GPU_TARGET}:${GPU_TARGET_FEATURES:-sramecc+:xnack-}"
)

if [ -d "$SPECIALIZED_OBJ_DIR" ] && [ "$(ls -A "$SPECIALIZED_OBJ_DIR"/*.o 2>/dev/null)" ]; then
    DEVICE_LINKER_ARGS+=(--input-dir "$SPECIALIZED_OBJ_DIR/")
    echo "  Including specialized kernels from: $SPECIALIZED_OBJ_DIR"
fi

# Note: onerank.cu is now compiled as part of dispatcher_combined.hip
# The oneRankReduce kernels and __clang_gpu_used_external are in the dispatcher ELF

$DEVICE_LINKER "${DEVICE_LINKER_ARGS[@]}"
echo "  Output: $MERGED_ELF ($(ls -lh "$MERGED_ELF" | awk '{print $5}'))"

# Step 4: Bundle merged device ELF
echo "Step 4: Bundling device code..."
$BUNDLER \
    -type=o \
    -bundle-align=4096 \
    -targets=host-x86_64-unknown-linux-gnu,$BUNDLE_TARGET \
    -input=/dev/null \
    -input="$MERGED_ELF" \
    -output="$FATBIN"

echo "  Output: $FATBIN ($(ls -lh "$FATBIN" | awk '{print $5}'))"

# Step 5: Host compilation with embedded fatbin
echo "Step 5: Compiling host code..."
$CLANG -cc1 \
    -triple x86_64-unknown-linux-gnu \
    -aux-triple amdgcn-amd-amdhsa \
    -O3 \
    -emit-obj \
    -mrelocation-model pic -pic-level 2 \
    -target-cpu x86-64 \
    -resource-dir "$CLANG_RESOURCE_DIR" \
    "${SYS_INCLUDES[@]}" \
    -include __clang_hip_runtime_wrapper.h \
    $DEFINES \
    "${INCLUDES[@]}" \
    -fcuda-include-gpubinary "$FATBIN" \
    -cuid=$CUID \
    -fcuda-allow-variadic-functions \
    -fhip-new-launch-api \
    -fgnuc-version=4.2.1 \
    -fcxx-exceptions -fexceptions \
    -o "$FINAL_OBJ" \
    -x hip "$SOURCE"

echo "  Output: $FINAL_OBJ ($(ls -lh "$FINAL_OBJ" | awk '{print $5}'))"

echo ""
echo "=== Build Complete ==="
echo "Final object: $FINAL_OBJ"

# Verify
echo ""
echo "=== Verification ==="
echo "Sections in final object:"
"$LLVM_PATH/bin/llvm-objdump" -h "$FINAL_OBJ" | grep -E "^\s+[0-9]+" | head -10
