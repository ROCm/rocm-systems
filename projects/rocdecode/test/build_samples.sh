#!/usr/bin/env bash
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# build_samples.sh — Build all rocDecode sample apps
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
JOBS="${1:-$(nproc)}"

for sample in rocdecDecode videoDecode videoDecodeBatch videoDecodeMem \
              videoDecodeMultiFiles videoDecodePerf videoDecodePicFiles \
              videoDecodeRaw videoDecodeRGB videoToSequence; do
  echo "=== Building $sample ==="
  mkdir -p "$PROJECT_ROOT/samples/${sample}/build"
  cmake -S "$PROJECT_ROOT/samples/${sample}" -B "$PROJECT_ROOT/samples/${sample}/build" 2>&1 | tail -2
  make clean -C "$PROJECT_ROOT/samples/${sample}/build" 2>&1 | tail -2
  make -j"$JOBS" -C "$PROJECT_ROOT/samples/${sample}/build" 2>&1 | tail -2
done

echo "=== All sample apps built ==="
