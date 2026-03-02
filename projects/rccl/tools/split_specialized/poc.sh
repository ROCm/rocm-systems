#!/bin/bash
#
# Proof of concept: split specialized kernel pipeline
#
# Compiles a handful of specialized kernels through:
#   source -> bitcode (-fgpu-rdc) -> assembly -> strip kernel -> assemble
# Then compiles the dispatcher (common.cu) through:
#   source -> bitcode -> assembly -> patch metadata -> assemble
# Links all device objects, bundles into hipfb, and creates host stub.
#
# Prerequisites:
#   - CMake configure must have run (for hipified sources + generated files)
#   - ROCm 7.x with hipcc, lld, clang-offload-bundler
#
# Usage:
#   cd <rccl-root>
#   tools/split_specialized/poc.sh [--dry-run]
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="$RCCL_ROOT/build/release"
HIPIFY="$BUILD_DIR/hipify"
ROCM="${ROCM_PATH:-/opt/rocm}"
CLANG="$ROCM/llvm/bin/clang"
LLD="$ROCM/llvm/bin/ld.lld"
BUNDLER="$ROCM/llvm/bin/clang-offload-bundler"
STRIP_KERNEL="$SCRIPT_DIR/strip_kernel.py"
PATCH_SCRIPT="$RCCL_ROOT/cmake/scripts/patch_kernel_metadata.cmake"

DRY_RUN=false
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=true

# Auto-detect GPU target from CMake cache
if [[ -f "$BUILD_DIR/CMakeCache.txt" ]]; then
    GPU=$(grep "^GPU_TARGETS:STRING=" "$BUILD_DIR/CMakeCache.txt" | sed 's/.*=//;s/;.*//')
else
    echo "ERROR: No CMakeCache.txt found. Run cmake configure first." >&2
    exit 1
fi
echo "GPU target: $GPU"

# Verify hipified sources exist
if [[ ! -f "$HIPIFY/src/device/common.h" ]]; then
    echo "ERROR: Hipified sources not found. Run 'ninja hipify_all' first." >&2
    exit 1
fi

# Verify device_table.h exists
if [[ ! -f "$HIPIFY/gensrc/device_table.h" ]]; then
    echo "ERROR: device_table.h not found. Run cmake configure first." >&2
    exit 1
fi

# Output directory
WORKDIR="$BUILD_DIR/split_specialized_poc"
rm -rf "$WORKDIR"
mkdir -p "$WORKDIR"/{bc,asm,meta,dev_obj,host_obj,fat_obj}

# Include flags (same as the CMake build)
INC_FLAGS=(
    -I"$HIPIFY/src"
    -I"$HIPIFY/src/device"
    -I"$HIPIFY/src/device/network/unpack"
    -I"$HIPIFY/src/include"
    -I"$HIPIFY/src/include/nccl_device"
    -I"$HIPIFY/src/include/plugin"
    -I"$HIPIFY/src/include/mlx5"
    -I"$HIPIFY/src/include/ionic"
    -I"$HIPIFY/gensrc"
    -I"$BUILD_DIR/include"
    -I"$ROCM/include"
)

# Compile definitions matching the CMake build
# Read ENABLE_* flags from CMakeCache
DEFS=(-DNDEBUG)
grep -q "FAULT_INJECTION:BOOL=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null && DEFS+=(-DENABLE_FAULT_INJECTION)
grep -q "LL128_ENABLED:.*=ON"     "$BUILD_DIR/CMakeCache.txt" 2>/dev/null && DEFS+=(-DENABLE_LL128)
grep -q "ENABLE_WARP_SPEED:.*=ON" "$BUILD_DIR/CMakeCache.txt" 2>/dev/null && DEFS+=(-DENABLE_WARP_SPEED)
grep -q "COLLTRACE:.*=ON"         "$BUILD_DIR/CMakeCache.txt" 2>/dev/null && DEFS+=(-DENABLE_COLLTRACE)
# NOTE: Do NOT define USE_INDIRECT_FUNCTION_CALL here.
# device_table.h (included by common.h when neither NCCL_SPECIALIZED_KERNEL nor
# DEVICE_LINKER is defined) provides table definitions and NCCL_CALL_FUNCTIONS_*
# dispatch. Defining USE_INDIRECT_FUNCTION_CALL triggers extern declarations in
# common.h with FUNC_COUNT=859, conflicting with device_table.h's actual sizes.
echo "Compile definitions: ${DEFS[*]}"

# Generate a small set of specialized kernels for the PoC
echo ""
echo "=== Generating specialized kernels ==="
SPEC_DIR="$WORKDIR/specialized_src"
mkdir -p "$SPEC_DIR"
python3 "$RCCL_ROOT/src/device/generate_specialized.py" "$SPEC_DIR" 0 "" "$GPU" 2>&1 | head -5

# Pick a representative subset (different collectives, types, protocols)
SPEC_FILES=(
    "$SPEC_DIR/specialized_all_reduce_ring_simple_sum_f32_unroll2.cpp"
    "$SPEC_DIR/specialized_all_reduce_ring_ll_sum_f32_unroll2.cpp"
    "$SPEC_DIR/specialized_all_gather_ring_simple_sum_i8_unroll2.cpp"
    "$SPEC_DIR/specialized_broadcast_ring_simple_sum_i8_unroll2.cpp"
    "$SPEC_DIR/specialized_reduce_scatter_ring_simple_sum_f32_unroll2.cpp"
)

# Verify all selected files exist
for f in "${SPEC_FILES[@]}"; do
    [[ -f "$f" ]] || { echo "ERROR: $f not found" >&2; exit 1; }
done
echo "Selected ${#SPEC_FILES[@]} specialized kernels for PoC"

# Common bc->asm flags
BC_FLAGS=(
    -x hip -std=c++17
    -fgpu-rdc
    --offload-device-only
    --offload-arch="$GPU"
    -emit-llvm -c -O3
    -fvisibility=hidden
    -Wno-unused-function
    -Wno-format-nonliteral
)

ASM_FLAGS=(
    -x ir
    -target amdgcn-amd-amdhsa
    "-mcpu=$GPU"
    -O3 -S
)

DEV_FLAGS=(
    -x assembler
    -target amdgcn-amd-amdhsa
    "-mcpu=$GPU"
    -c
)

echo ""
echo "=== Compiling specialized kernels ==="
SPEC_DEV_OBJS=()
SPEC_META_FILES=()

for src in "${SPEC_FILES[@]}"; do
    fname=$(basename "$src" .cpp)
    bc="$WORKDIR/bc/$fname.bc"
    asm="$WORKDIR/asm/$fname.s"
    stripped="$WORKDIR/asm/$fname.stripped.s"
    meta="$WORKDIR/meta/$fname.meta"
    obj="$WORKDIR/dev_obj/$fname.o"

    echo "  $fname"

    # Step A: source -> bitcode
    hipcc "${BC_FLAGS[@]}" -DNCCL_SPECIALIZED_KERNEL=1 "${DEFS[@]}" "${INC_FLAGS[@]}" -o "$bc" "$src" 2>&1
    echo "    [bc] done"

    # Step B1: bitcode -> assembly
    "$CLANG" "${ASM_FLAGS[@]}" -o "$asm" "$bc" 2>&1
    echo "    [asm] done ($(wc -l < "$asm") lines)"

    # Step B2: strip kernel + extract metadata
    python3 "$STRIP_KERNEL" "$asm" "$stripped" --meta "$meta" 2>&1 | sed 's/^/    /'

    # Step B3: assemble stripped .s -> .o
    "$CLANG" "${DEV_FLAGS[@]}" -o "$obj" "$stripped" 2>&1
    echo "    [dev] done"

    SPEC_DEV_OBJS+=("$obj")
    SPEC_META_FILES+=("$meta")
done

echo ""
echo "=== Compiling dispatcher (common.cu) ==="
# The dispatcher includes device_table.h which references all ncclDevFunc_* symbols.
# With -fgpu-rdc these become relocations resolved by lld.
DISP_SRC="$HIPIFY/src/device/common.cu.cpp"
DISP_BC="$WORKDIR/bc/common.bc"
DISP_ASM="$WORKDIR/asm/common.s"
DISP_PATCHED="$WORKDIR/asm/common.patched.s"
DISP_OBJ="$WORKDIR/dev_obj/common.o"

# Compile dispatcher to bitcode (NOT a specialized kernel, so no NCCL_SPECIALIZED_KERNEL)
hipcc "${BC_FLAGS[@]}" "${DEFS[@]}" "${INC_FLAGS[@]}" -o "$DISP_BC" "$DISP_SRC" 2>&1
echo "  [bc] done"

# Bitcode -> assembly
"$CLANG" "${ASM_FLAGS[@]}" -o "$DISP_ASM" "$DISP_BC" 2>&1
echo "  [asm] done ($(wc -l < "$DISP_ASM") lines)"

# Safety: verify uses_dynamic_stack is present
if ! grep -q "uses_dynamic_stack: true" "$DISP_ASM"; then
    echo "FATAL: Dispatcher assembly missing .uses_dynamic_stack: true" >&2
    exit 1
fi
echo "  [check] uses_dynamic_stack: true confirmed"

# Report dispatcher LDS (only the dispatcher's LDS matters; specialized kernels
# use extern __shared__ so their LDS is smaller and irrelevant)
disp_lds=$(grep -o 'group_segment_fixed_size: [0-9]*' "$DISP_ASM" | head -1 | grep -o '[0-9]*')
echo "  [check] Dispatcher LDS: $disp_lds"

# Patch dispatcher metadata using callee sidecars
MANIFEST="$WORKDIR/meta/manifest.txt"
printf '%s\n' "${SPEC_META_FILES[@]}" > "$MANIFEST"
cp "$DISP_ASM" "$DISP_PATCHED"

# Check if we have patch_kernel_metadata.cmake from the prototype
if [[ -f "$PATCH_SCRIPT" ]]; then
    cmake -DASM_FILE="$DISP_PATCHED" -DMANIFEST="$MANIFEST" -P "$PATCH_SCRIPT" 2>&1 | sed 's/^/  /'
    echo "  [patch] done"
else
    echo "  [patch] SKIPPED (patch_kernel_metadata.cmake not found at $PATCH_SCRIPT)"
    echo "          The dispatcher metadata will have kernel-only register counts."
    echo "          Copy cmake/scripts/patch_kernel_metadata.cmake from the prototype."
fi

# Assemble patched dispatcher
"$CLANG" "${DEV_FLAGS[@]}" -o "$DISP_OBJ" "$DISP_PATCHED" 2>&1
echo "  [dev] done"

echo ""
echo "=== Linking device objects ==="
ALL_OBJS=("${SPEC_DEV_OBJS[@]}" "$DISP_OBJ")
COMBINED_O="$WORKDIR/dev_obj/combined.o"
COMBINED_SO="$WORKDIR/dev_obj/combined.so"

"$LLD" -r -o "$COMBINED_O" "${ALL_OBJS[@]}" 2>&1
echo "  [lld -r] done: $(du -h "$COMBINED_O" | cut -f1)"

"$LLD" -shared -o "$COMBINED_SO" "$COMBINED_O" 2>&1
echo "  [lld -shared] done: $(du -h "$COMBINED_SO" | cut -f1)"

# Verify ncclDevFunc symbols in combined object
echo ""
echo "=== Verification ==="
SYMTAB=$(/opt/rocm/llvm/bin/llvm-readelf --symbols "$COMBINED_SO" 2>/dev/null)

echo "  Defined ncclDevFunc symbols:"
echo "$SYMTAB" | grep "ncclDevFunc_" | grep -v " UND " || true

echo ""
echo "  Undefined ncclDevFunc symbols (expected for non-compiled kernels):"
UNDEF_COUNT=$(echo "$SYMTAB" | grep "ncclDevFunc_" | grep -c " UND " || true)
echo "  $UNDEF_COUNT undefined ncclDevFunc references (from device_table.h)"

echo ""
echo "  ncclDevKernel symbols (should be dispatcher only):"
echo "$SYMTAB" | grep "ncclDevKernel" || true

echo ""
echo "  ncclDevFuncTable symbols:"
echo "$SYMTAB" | grep "ncclDevFuncTable" || true

echo ""
echo "=== Bundling ==="
HIPFB="$WORKDIR/fat_obj/combined.hipfb"
BUNDLER_DEVICE="hipv4-amdgcn-amd-amdhsa--$GPU"
BUNDLER_HOST="host-x86_64-unknown-linux-gnu"

"$BUNDLER" \
    --type=bc \
    "--targets=$BUNDLER_HOST,$BUNDLER_DEVICE" \
    --input=/dev/null \
    "--input=$COMBINED_SO" \
    "--output=$HIPFB" 2>&1
echo "  [hipfb] done: $(du -h "$HIPFB" | cut -f1)"

echo ""
echo "=== Host stub compilation ==="
HOST_OBJ="$WORKDIR/host_obj/common.host.o"

hipcc \
    -x hip -std=c++17 \
    --offload-host-only \
    --offload-arch="$GPU" \
    -Xclang -fcuda-include-gpubinary \
    -Xclang "$HIPFB" \
    -c -O3 -fPIC \
    "${DEFS[@]}" \
    "${INC_FLAGS[@]}" \
    -fvisibility=hidden \
    -Wno-unused-function \
    -Wno-format-nonliteral \
    -o "$HOST_OBJ" "$DISP_SRC" 2>&1
echo "  [host] done: $(du -h "$HOST_OBJ" | cut -f1)"

# Verify .hip_fatbin section in host object
echo ""
echo "=== Final verification ==="
echo "  Sections in host stub:"
/opt/rocm/llvm/bin/llvm-readelf -S "$HOST_OBJ" 2>/dev/null | grep -i "fatbin\|hip"

echo ""
echo "=== DONE ==="
echo "Output directory: $WORKDIR"
echo ""
echo "Key files:"
echo "  Device code object: $COMBINED_SO"
echo "  Fat binary blob:    $HIPFB"
echo "  Host stub object:   $HOST_OBJ"
