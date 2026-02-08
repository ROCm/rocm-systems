#!/usr/bin/env bash
# Integration test: build with -l --device-linker, then disassemble merged device ELF
# with line/source info. Compares first N lines against golden to catch DWARF/merge regressions.
#
# Usage:
#   From RCCL root, after a full build:
#     tools/device_linker/run_disasm_integration_test.sh
#   Or set BUILD_DIR to point at your build (e.g. build/release).
#
# Build command (run once from repo root):
#   rm -rf build && ./install.sh -l --device-linker

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RCCL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$RCCL_ROOT/build/release}"
MERGED_ELF="$BUILD_DIR/device_linker_output/merged_device.elf"
LLVM_OBJDUMP="${LLVM_OBJDUMP:-/opt/rocm/llvm/bin/llvm-objdump}"
GOLDEN_LINES=100000
GOLDEN_FILE="$SCRIPT_DIR/merged_device_disasm_golden.txt"
OUT_DISASM="$SCRIPT_DIR/merged_device_disasm.txt"

if [[ ! -f "$MERGED_ELF" ]]; then
  echo "Missing merged ELF: $MERGED_ELF"
  echo "Run from repo root: rm -rf build && ./install.sh -l --device-linker"
  exit 1
fi

if [[ ! -x "$LLVM_OBJDUMP" ]]; then
  echo "llvm-objdump not found: $LLVM_OBJDUMP (set LLVM_OBJDUMP to override)"
  exit 1
fi

echo "Disassembling (with -l -d --source): $MERGED_ELF"
"$LLVM_OBJDUMP" -l -d --source "$MERGED_ELF" > "$OUT_DISASM" 2>/dev/null || true

if [[ ! -f "$GOLDEN_FILE" ]]; then
  echo "No golden file yet: $GOLDEN_FILE"
  echo "Create with: head -${GOLDEN_LINES} $OUT_DISASM > $GOLDEN_FILE"
  exit 1
fi

echo "Comparing first $GOLDEN_LINES lines to golden..."
# Normalize path line (llvm-objdump prints input path on line 2; use placeholder so test is path-independent)
normalize() { head -"$GOLDEN_LINES" "$1" | sed 's|.*/merged_device\.elf:|merged_device.elf:|'; }
diff -u <(normalize "$GOLDEN_FILE") <(normalize "$OUT_DISASM") && echo "PASS: disassembly matches golden." || { echo "FAIL: disassembly differs from golden."; exit 1; }
