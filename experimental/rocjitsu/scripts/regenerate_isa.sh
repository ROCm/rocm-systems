#!/bin/bash
# Copyright (c) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT

# Regenerate all AMDGPU ISA C++ files from Machine-Readable ISA XML specs.
# Uses --multi mode to run CrossIsaAnalyzer and generate shared execute headers.
#
# Prerequisites:
#   - Python 3.10+ with the amdisa package (lib/python/amdisa/) installed
#   - MR ISA XMLs present under third_party/machine-readable-isa/
#     (run: python scripts/download_machine_readable_isa.py --force)
#
# Usage:
#   ./scripts/regenerate_isa.sh
#
# After regeneration, clang-format is automatically applied to the output.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

MRISA="$PROJECT_ROOT/third_party/machine-readable-isa"
OUT="$PROJECT_ROOT/lib/rocjitsu/src/rocjitsu/isa/arch/amdgpu"

# Verify the XML sources exist.
for arch in cdna1 cdna2 cdna3 cdna4 rdna1 rdna2 rdna3 rdna3_5 rdna4; do
  xml="$MRISA/amdgpu_isa_${arch}.xml"
  if [ ! -f "$xml" ]; then
    echo "error: missing $xml" >&2
    echo "Run: python scripts/download_machine_readable_isa.py --force" >&2
    exit 1
  fi
done

echo "Regenerating ISA C++ files..."

PYTHONPATH=lib/python python -m amdisa --multi \
  cdna1:"$MRISA/amdgpu_isa_cdna1.xml" \
  cdna2:"$MRISA/amdgpu_isa_cdna2.xml" \
  cdna3:"$MRISA/amdgpu_isa_cdna3.xml" \
  cdna4:"$MRISA/amdgpu_isa_cdna4.xml" \
  rdna1:"$MRISA/amdgpu_isa_rdna1.xml" \
  rdna2:"$MRISA/amdgpu_isa_rdna2.xml" \
  rdna3:"$MRISA/amdgpu_isa_rdna3.xml" \
  rdna3_5:"$MRISA/amdgpu_isa_rdna3_5.xml" \
  rdna4:"$MRISA/amdgpu_isa_rdna4.xml" \
  --gen-all --gen-shared-execute -o "$OUT"

echo "Running clang-format on generated files..."
bash "$SCRIPT_DIR/clang_format.sh" "$OUT"

echo "Done. Regenerated ISA files in: $OUT"
