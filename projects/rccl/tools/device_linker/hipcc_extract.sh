#!/bin/bash
# Wrapper around hipcc that extracts device code after compilation.
# Usage: hipcc_extract.sh [hipcc args...]
#
# For compilation of specialized kernels (-c flag, output is .o):
#   - Removes --offload-compress
#   - Compiles with hipcc
#   - Extracts device code to .device.o alongside the .o
#
# For linking, version checks, or other operations: passes through unchanged.

# Use HIPCC_REAL if set, otherwise default to hipcc
HIPCC="${HIPCC_REAL:-/opt/rocm/bin/hipcc}"

# For version queries, just pass through
if [[ "$1" == "--version" ]] || [[ "$1" == "-v" ]] || [[ "$1" == "-V" ]]; then
    exec "$HIPCC" "$@"
fi

# Check if this is a compilation (-c flag present)
IS_COMPILE=0
OUTPUT_FILE=""
ARGS=()
SKIP_NEXT=0

for arg in "$@"; do
    if [[ $SKIP_NEXT -eq 1 ]]; then
        OUTPUT_FILE="$arg"
        SKIP_NEXT=0
        ARGS+=("$arg")
        continue
    fi
    
    if [[ "$arg" == "-c" ]]; then
        IS_COMPILE=1
    fi
    
    if [[ "$arg" == "-o" ]]; then
        SKIP_NEXT=1
        ARGS+=("$arg")
        continue
    fi
    
    # Remove --offload-compress for compilations
    if [[ "$arg" == "--offload-compress" ]] && [[ $IS_COMPILE -eq 1 ]]; then
        continue
    fi
    
    ARGS+=("$arg")
done

# Run the real compiler with (possibly modified) arguments
"$HIPCC" "${ARGS[@]}"
HIPCC_RC=$?

if [[ $HIPCC_RC -ne 0 ]]; then
    exit $HIPCC_RC
fi

# Only extract for compilation of specialized kernels
if [[ $IS_COMPILE -eq 1 ]] && [[ -n "$OUTPUT_FILE" ]] && [[ "$OUTPUT_FILE" == *.o ]]; then
    # Check if this is a specialized kernel
    if [[ "$OUTPUT_FILE" == *specialized_* ]]; then
        FATBIN_TMP="${OUTPUT_FILE}.fatbin.tmp"
        DEVICE_OUT="${OUTPUT_FILE%.o}.device.o"
        
        # Extract .hip_fatbin section
        /opt/rocm/llvm/bin/llvm-objcopy --dump-section=.hip_fatbin="$FATBIN_TMP" "$OUTPUT_FILE" 2>/dev/null
        
        if [[ -f "$FATBIN_TMP" ]]; then
            # Unbundle to get device code for gfx942
            /opt/rocm/llvm/bin/clang-offload-bundler --type=o \
                --targets=hipv4-amdgcn-amd-amdhsa--gfx942 \
                --input="$FATBIN_TMP" \
                --output="$DEVICE_OUT" \
                --unbundle 2>/dev/null
            
            rm -f "$FATBIN_TMP"
        fi
    fi
fi

exit 0
