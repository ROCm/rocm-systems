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
GPU_TARGET_FEATURES=$(echo "$GPU_TARGET_FULL" | sed 's/^[^:]*://' | tr ':' '\n' | sort | tr '\n' ':' | sed 's/:$//')
if [ -n "$GPU_TARGET_FEATURES" ]; then
    BUNDLE_TARGET="hipv4-amdgcn-amd-amdhsa--${GPU_TARGET}:${GPU_TARGET_FEATURES}"
else
    BUNDLE_TARGET="hipv4-amdgcn-amd-amdhsa--${GPU_TARGET}"
fi

# Output directory - use build directory
OUTPUT_DIR="$BUILD_DIR/device_linker_output"
mkdir -p "$OUTPUT_DIR"

# Tools
CLANG=/opt/rocm/lib/llvm/bin/clang-22
LLD=/opt/rocm/lib/llvm/bin/lld
BUNDLER=/opt/rocm/lib/llvm/bin/clang-offload-bundler
DEVICE_LINKER="$SCRIPT_DIR/device_linker"

# BUNDLE_TARGET is set above based on GPU_TARGET_FULL

# Source and output files
SOURCE="$HIPIFY_DIR/src/device/common.cu.cpp"
DEVICE_OBJ="$OUTPUT_DIR/dispatcher_device.o"
DEVICE_ELF="$OUTPUT_DIR/dispatcher_device.elf"
MERGED_ELF="$OUTPUT_DIR/merged_device.elf"
FATBIN="$OUTPUT_DIR/merged.hipfb"
FINAL_OBJ="$OUTPUT_DIR/dispatcher_final.o"

# Specialized kernel directory (from CMake build)
SPECIALIZED_OBJ_DIR="$BUILD_DIR/specialized_objs"

# Onerank device object (compiled separately)
ONERANK_OBJ="$OUTPUT_DIR/onerank_device.o"

# Host table for funcId mapping
HOST_TABLE="$HIPIFY_DIR/gensrc/host_table.cpp"

# Check prerequisites
if [ ! -f "$SOURCE" ]; then
    echo "Error: Source not found: $SOURCE"
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
DEFINES="-DDEVICE_LINKER -DDEVICE_LINKER_DISPATCH -DENABLE_FAULT_INJECTION"

# CUID (compilation unit ID) - needs to match between device and host compilation
CUID="devicelinker$(echo -n "$SOURCE" | md5sum | cut -c1-16)"

echo "=== Building with Device Linker (Option B) ==="
echo "Source: $SOURCE"
echo "GPU Target: $GPU_TARGET"
echo "CUID: $CUID"
echo ""

# System include paths (from hipcc -### output)
SYS_INCLUDES=(
    "-internal-isystem" "/opt/rocm/lib/llvm/lib/clang/22/include/cuda_wrappers"
    "-idirafter" "/opt/rocm/include"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/x86_64-linux-gnu/c++/12"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../include/c++/12/backward"
    "-internal-isystem" "/opt/rocm/lib/llvm/lib/clang/22/include"
    "-internal-isystem" "/usr/local/include"
    "-internal-isystem" "/usr/lib/gcc/x86_64-linux-gnu/12/../../../../x86_64-linux-gnu/include"
    "-internal-externc-isystem" "/usr/include/x86_64-linux-gnu"
    "-internal-externc-isystem" "/include"
    "-internal-externc-isystem" "/usr/include"
)

# Step 1: Compile dispatcher device code
echo "Step 1: Compiling device code..."
$CLANG -cc1 \
    -triple amdgcn-amd-amdhsa \
    -aux-triple x86_64-unknown-linux-gnu \
    -O3 \
    -emit-obj \
    -fcuda-is-device \
    -fno-threadsafe-statics \
    -mllvm -amdgpu-internalize-symbols \
    -fvisibility=hidden \
    -fapply-global-visibility-to-externs \
    -aux-target-cpu x86-64 \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/ocml.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/ockl.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_daz_opt_off.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_unsafe_math_off.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_finite_only_off.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_correctly_rounded_sqrt_on.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_wavefrontsize64_on.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_isa_version_942.bc \
    -mlink-builtin-bitcode /opt/rocm/lib/llvm/lib/clang/22/lib/amdgcn/bitcode/oclc_abi_version_600.bc \
    -target-cpu $GPU_TARGET \
    -resource-dir /opt/rocm/lib/llvm/lib/clang/22 \
    "${SYS_INCLUDES[@]}" \
    -include __clang_hip_runtime_wrapper.h \
    -D DEVICE_LINKER -D DEVICE_LINKER_DISPATCH -D ENABLE_FAULT_INJECTION \
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
    --target "${GPU_TARGET}:${GPU_TARGET_FEATURES:-sramecc+:xnack+}"
)

if [ -d "$SPECIALIZED_OBJ_DIR" ] && [ "$(ls -A "$SPECIALIZED_OBJ_DIR"/*.o 2>/dev/null)" ]; then
    DEVICE_LINKER_ARGS+=(--input-dir "$SPECIALIZED_OBJ_DIR/")
    echo "  Including specialized kernels from: $SPECIALIZED_OBJ_DIR"
fi

if [ -f "$ONERANK_OBJ" ]; then
    DEVICE_LINKER_ARGS+=(--onerank "$ONERANK_OBJ")
    echo "  Including onerank device code from: $ONERANK_OBJ"
fi

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
    -resource-dir /opt/rocm/lib/llvm/lib/clang/22 \
    "${SYS_INCLUDES[@]}" \
    -include __clang_hip_runtime_wrapper.h \
    -D DEVICE_LINKER -D DEVICE_LINKER_DISPATCH -D ENABLE_FAULT_INJECTION \
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
/opt/rocm/llvm/bin/llvm-objdump -h "$FINAL_OBJ" | grep -E "^\s+[0-9]+" | head -10
