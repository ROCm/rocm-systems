#!/usr/bin/env bash
#
# disassemble.sh - Extract and disassemble GPU code from HIP fat binaries,
# or disassemble raw HSA code object ELFs directly (e.g. the
# <pid>_<gfxN>_code_object_id_<N>.out snapshots dumped by rocprofiler-sdk-tool,
# which are already unbundled AMDGPU ELFs, not clang offload bundles).
#
# Works without a full ROCm install. Only requires:
#   - objcopy, python3 (for fat-binary extraction)
#   - llvm-objdump from ROCm (for disassembly)
#
# Usage:
#   ./disassemble.sh <hip_binary_or_code_object> [options]
#
# Options:
#   -o, --output FILE       Write disassembly to FILE instead of stdout
#   -k, --keep              Keep the extracted .hsaco file (default: deleted)
#   -t, --target TARGET     GPU target to extract/disassemble (default: gfx1310)
#   --raw                   Treat INPUT as an already-unbundled code object ELF
#                           (skips .hip_fatbin extraction/unbundling entirely)
#   --llvm-objdump PATH     Path to llvm-objdump (auto-detected if not set)
#   --list                  List embedded GPU targets and exit (fat binary only)
#   -h, --help              Show this help message
#
# Examples:
#   ./disassemble.sh ../tmp_run_vlaindic/dummy_work
#   ./disassemble.sh ../tmp_run_vlaindic/dummy_work -o dump.s -k
#   ./disassemble.sh ../tmp_run_vlaindic/dummy_work --list
#   ./disassemble.sh ../tmp_run_vlaindic/dummy_work -t gfx942
#   ./disassemble.sh build/.../115499_gfx1250_code_object_id_1.out --raw -t gfx1250

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SUITE_DIR="$(dirname "$SCRIPT_DIR")"

# Defaults
TARGET="gfx1310"
OUTPUT=""
KEEP_HSACO=false
LIST_ONLY=false
LLVM_OBJDUMP=""
RAW_MODE=false

# --- Helpers ----------------------------------------------------------------

usage() {
    sed -n '2,/^$/{ s/^# \?//; p }' "$0"
    exit 0
}

die() { echo "ERROR: $*" >&2; exit 1; }

# Auto-detect llvm-objdump from the local ROCm copy
find_llvm_objdump() {
    if [[ -n "$LLVM_OBJDUMP" ]]; then
        return
    fi

    # Check common locations relative to this repo / known install paths
    local candidates=(
        "$SUITE_DIR/../opt/rocm-7.14.0-gfx1310/lib/llvm/bin/llvm-objdump"
        "$HOME/work/opt/rocm-7.14.0-gfx1310/lib/llvm/bin/llvm-objdump"
        # /opt/rocm is conventionally a symlink to the canonical current install
        "/opt/rocm/lib/llvm/bin/llvm-objdump"
    )

    # Also try any rocm install under ~/work/opt or /opt
    while IFS= read -r -d '' f; do
        candidates+=("$f")
    done < <(find "$HOME/work/opt" /opt -maxdepth 6 -path "*/llvm/bin/llvm-objdump" -type f -print0 2>/dev/null || true)

    # Fallback: system PATH (may be too old to support newer GPU targets)
    candidates+=("$(command -v llvm-objdump 2>/dev/null || true)")

    for c in "${candidates[@]}"; do
        if [[ -n "$c" && -x "$c" ]]; then
            LLVM_OBJDUMP="$c"
            return
        fi
    done

    die "Could not find llvm-objdump. Use --llvm-objdump to specify the path."
}

# Python helper: parse __CLANG_OFFLOAD_BUNDLE__ and either list targets or
# extract a specific one.
extract_gpu_code() {
    local fatbin_file="$1"
    local action="$2"   # "list" or "extract"
    local target="$3"
    local out_file="$4"

    python3 - "$fatbin_file" "$action" "$target" "$out_file" <<'PYEOF'
import struct, sys

fatbin_path, action, target, out_path = sys.argv[1:5]

with open(fatbin_path, "rb") as f:
    data = f.read()

magic = data[:24]
if magic != b"__CLANG_OFFLOAD_BUNDLE__":
    print(f"ERROR: Not a clang offload bundle (magic: {magic!r})", file=sys.stderr)
    sys.exit(1)

num_bundles = struct.unpack_from("<Q", data, 24)[0]
offset = 32
bundles = []

for i in range(num_bundles):
    bundle_offset, bundle_size, triple_size = struct.unpack_from("<QQQ", data, offset)
    triple = data[offset + 24 : offset + 24 + triple_size].decode("ascii")
    bundles.append((triple, bundle_offset, bundle_size))
    offset += 24 + triple_size

if action == "list":
    for triple, bo, bs in bundles:
        print(f"  {triple}  (offset={bo}, size={bs})")
    sys.exit(0)

# Extract
for triple, bo, bs in bundles:
    if target in triple:
        if bs == 0:
            print(f"ERROR: Bundle for '{triple}' has size 0", file=sys.stderr)
            sys.exit(1)
        with open(out_path, "wb") as out:
            out.write(data[bo : bo + bs])
        print(f"Extracted {bs} bytes from \"{triple}\" -> {out_path}", file=sys.stderr)
        sys.exit(0)

print(f"ERROR: No bundle matching '{target}' found. Available targets:", file=sys.stderr)
for triple, _, _ in bundles:
    print(f"  {triple}", file=sys.stderr)
sys.exit(1)
PYEOF
}

# --- Argument parsing -------------------------------------------------------

INPUT=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)    usage ;;
        -o|--output)  OUTPUT="$2"; shift 2 ;;
        -k|--keep)    KEEP_HSACO=true; shift ;;
        -t|--target)  TARGET="$2"; shift 2 ;;
        --llvm-objdump) LLVM_OBJDUMP="$2"; shift 2 ;;
        --list)       LIST_ONLY=true; shift ;;
        --raw)        RAW_MODE=true; shift ;;
        -*)           die "Unknown option: $1" ;;
        *)
            [[ -z "$INPUT" ]] || die "Multiple input files not supported"
            INPUT="$1"; shift ;;
    esac
done

[[ -n "$INPUT" ]] || die "No input file specified. Run with --help for usage."
[[ -f "$INPUT" ]] || die "File not found: $INPUT"

# --- Main -------------------------------------------------------------------

TMPDIR_WORK="$(mktemp -d)"
trap 'rm -rf "$TMPDIR_WORK"' EXIT

# Auto-detect: a raw HSA code object (e.g. the <pid>_<gfxN>_code_object_id_<N>.out
# snapshots dumped by rocprofiler-sdk-tool) is already an unbundled AMDGPU ELF,
# not a clang offload bundle. Peek at the magic to tell them apart, unless the
# caller already forced --raw.
# Auto-detect raw vs. fat-binary input. Both a host HIP binary and a raw
# AMDGPU code object snapshot are plain ELF files at offset 0 (the fatbin is
# only a *section* inside the host ELF), so the file magic alone can't tell
# them apart. Instead check the ELF machine type: a raw code object snapshot
# targets "AMD GPU" directly, while a host binary targets the host's own ISA
# (e.g. x86-64) and carries the device code inside its .hip_fatbin section.
if ! $RAW_MODE; then
    if readelf -h "$INPUT" 2>/dev/null | grep -q 'Machine:.*AMD GPU'; then
        RAW_MODE=true
    elif ! readelf -h "$INPUT" >/dev/null 2>&1; then
        die "'$INPUT' is not a valid ELF file."
    fi
fi

if $RAW_MODE; then
    $LIST_ONLY && die "--list is only supported for HIP fat binaries, not raw code objects."
    HSACO="$INPUT"
else
    FATBIN="$TMPDIR_WORK/hip_fatbin.bin"

    # Step 1: Extract .hip_fatbin section
    if ! objcopy -O binary --only-section=.hip_fatbin "$INPUT" "$FATBIN" 2>/dev/null; then
        die "Failed to extract .hip_fatbin section. Is this a HIP binary?"
    fi

    if [[ ! -s "$FATBIN" ]]; then
        die "No .hip_fatbin section found in '$INPUT'. Is this a HIP binary?"
    fi

    # Step 2: List or extract
    if $LIST_ONLY; then
        echo "Embedded GPU targets in $(basename "$INPUT"):"
        extract_gpu_code "$FATBIN" "list" "" ""
        exit 0
    fi

    HSACO="$TMPDIR_WORK/${TARGET}.hsaco"
    extract_gpu_code "$FATBIN" "extract" "$TARGET" "$HSACO"

    # Optionally keep the hsaco
    if $KEEP_HSACO; then
        KEEP_PATH="${INPUT%.out}.${TARGET}.hsaco"
        [[ "$KEEP_PATH" == "$INPUT" ]] && KEEP_PATH="${INPUT}.${TARGET}.hsaco"
        cp "$HSACO" "$KEEP_PATH"
        echo "Saved code object: $KEEP_PATH" >&2
    fi
fi

# Step 3: Disassemble
find_llvm_objdump

echo "Using: $LLVM_OBJDUMP" >&2

if [[ -n "$OUTPUT" ]]; then
    "$LLVM_OBJDUMP" -l -d --mcpu="$TARGET" "$HSACO" > "$OUTPUT"
    echo "Disassembly written to: $OUTPUT" >&2
else
    "$LLVM_OBJDUMP" -l -d --mcpu="$TARGET" "$HSACO"
fi
