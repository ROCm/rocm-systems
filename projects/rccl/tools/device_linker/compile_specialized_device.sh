#!/bin/bash
#
# Compile a single specialized kernel source to a device-only object (.device.o).
# Uses the same clang -cc1 device compilation as the dispatcher (no host fat binary).
#
# Usage: compile_specialized_device.sh BUILD_DIR GPU_TARGET SOURCE_FILE OUTPUT_FILE
#   BUILD_DIR: CMake build directory (e.g. build/release)
#   GPU_TARGET: Full GPU target (e.g. gfx942 or gfx942:xnack+:sramecc+)
#   SOURCE_FILE: Path to the .cpp specialized kernel source
#   OUTPUT_FILE: Path for the output device object (e.g. name.device.o)
#
# Environment (from CMake): ENABLE_FAULT_INJECTION, ENABLE_WARP_SPEED, ENABLE_LL128,
#   DEBUG_PEER_POINTERS, ENABLE_COLLTRACE, ENABLE_PROFILING (0/1)
#

set -e

if [ $# -lt 4 ]; then
    echo "Usage: $0 BUILD_DIR GPU_TARGET SOURCE_FILE OUTPUT_FILE"
    exit 1
fi

BUILD_DIR="$1"
GPU_TARGET_FULL="$2"
SOURCE_FILE="$3"
OUTPUT_FILE="$4"

# Strip feature flags for -target-cpu
GPU_TARGET=$(echo "$GPU_TARGET_FULL" | sed 's/:.*$//')
HIPIFY_DIR="$BUILD_DIR/hipify"

# Tools - same as build_with_device_linker.sh
ROCM_PATH=${ROCM_PATH:-/opt/rocm}
AMDCLANG="$ROCM_PATH/bin/amdclang++"
CLANG_RESOURCE_DIR=$($AMDCLANG -print-resource-dir)
LLVM_PATH=$(dirname $(dirname $(dirname "$CLANG_RESOURCE_DIR")))
CLANG="$LLVM_PATH/bin/clang"
if [ -d "$CLANG_RESOURCE_DIR/lib/amdgcn/bitcode" ]; then
    BITCODE_DIR="$CLANG_RESOURCE_DIR/lib/amdgcn/bitcode"
else
    BITCODE_DIR=$(dirname $(find "$LLVM_PATH/lib/clang" -name "ocml.bc" 2>/dev/null | head -1))
fi
ISA_VERSION=$(echo "$GPU_TARGET" | sed 's/gfx//')

# Include paths - match main CMakeLists.txt specialized kernel includes
INCLUDES=(
    "-I$HIPIFY_DIR/src"
    "-I$HIPIFY_DIR/src/include"
    "-I$HIPIFY_DIR/src/include/nccl_device"
    "-I$HIPIFY_DIR/src/include/plugin"
    "-I$HIPIFY_DIR/src/device"
    "-I$HIPIFY_DIR/gensrc"
    "-I$BUILD_DIR/include"
)

# Defines - must match host and dispatcher (structure layout)
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
if [ "${DEBUG_PEER_POINTERS:-0}" = "1" ]; then
    DEFINES="$DEFINES -DDEBUG_PEER_POINTERS"
fi
if [ "${ENABLE_COLLTRACE:-1}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_COLLTRACE"
fi
if [ "${ENABLE_PROFILING:-0}" = "1" ]; then
    DEFINES="$DEFINES -DENABLE_PROFILING"
fi

# System include paths
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

# Compile to device object only (no host, no fat binary)
$CLANG -cc1 \
    -triple amdgcn-amd-amdhsa \
    -aux-triple x86_64-unknown-linux-gnu \
    -O3 \
    -emit-obj \
    -debug-info-kind=line-tables-only \
    -dwarf-version=5 \
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
    -fcuda-allow-variadic-functions \
    -o "$OUTPUT_FILE" \
    -x hip "$SOURCE_FILE"
