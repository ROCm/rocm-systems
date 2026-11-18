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
# build_samples.sh — Build all rocDecode sample apps.
#
# Runs as a bash script on both OSes: natively on Linux, and under Git Bash on
# Windows. The Windows build toolchain is still MSVC — cmake auto-selects the
# Visual Studio generator (multi-config), which is why the Windows path adds
# --config Release. The shell is Git Bash; the compiler is MSVC.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
# Default the job count to the core count, but don't let a missing nproc abort the
# script under `set -e`: fall back to Windows' NUMBER_OF_PROCESSORS, then to 4. An
# explicit first arg always wins.
JOBS="${1:-$(nproc 2>/dev/null || echo "${NUMBER_OF_PROCESSORS:-4}")}"

# Under Git Bash on Windows, cmake auto-selects the MSVC multi-config (Visual
# Studio) generator, so the build step needs an explicit --config Release; the
# Linux single-config Makefile generator does not. CONFIG_ARGS stays empty on
# Linux (preserving the original invocation) and carries the flag on Windows
# (Git Bash reports MINGW*/MSYS*/CYGWIN* from uname). On Windows, ffmpeg's bin must
# be on PATH before running; the samples' FindFFmpeg.cmake auto-discovers it there.
CONFIG_ARGS=()
case "$(uname -s)" in
  MINGW*|MSYS*|CYGWIN*) CONFIG_ARGS=(--config Release) ;;
esac

# Fail fast if ROCM_PATH does not point at a usable ROCm toolchain (samples derive
# their compiler from it; a wrong/unset value fails confusingly). On Windows the
# compiler is MSVC (no amdclang++), so only verify the install prefix exists.
if [[ ${#CONFIG_ARGS[@]} -eq 0 ]]; then
  ROCM_PATH_EFF="${ROCM_PATH:-/opt/rocm}"
  if [[ ! -x "$ROCM_PATH_EFF/lib/llvm/bin/amdclang++" ]]; then
    echo "ERROR: ROCm toolchain not found at: $ROCM_PATH_EFF/lib/llvm/bin/amdclang++" >&2
    echo "       Set ROCM_PATH to your ROCm install. Skills run cmake in a non-interactive" >&2
    echo "       shell (no ~/.bashrc), so set ROCM_PATH in ~/.profile so the build sees it." >&2
    exit 1
  fi
else
  ROCM_PATH_EFF="${ROCM_PATH:-C:/opt/rocm}"
  if [[ ! -d "$ROCM_PATH_EFF" ]]; then
    echo "ERROR: ROCM_PATH does not exist: $ROCM_PATH_EFF" >&2
    echo "       Set ROCM_PATH to your rocDecode install prefix (default C:/opt/rocm)" >&2
    echo "       in the env you set up before running this." >&2
    exit 1
  fi
fi

# Export the effective prefix so the per-sample cmake child processes inherit it.
# Without this, an unset ROCM_PATH that we defaulted above is invisible to cmake,
# and the samples' CMakeLists.txt would fall back to its own /opt/rocm default and
# resolve dependencies from the wrong install tree.
export ROCM_PATH="$ROCM_PATH_EFF"

for sample in rocdecDecode videoDecode videoDecodeBatch videoDecodeMem \
              videoDecodeMultiFiles videoDecodePerf videoDecodePicFiles \
              videoDecodeRaw videoDecodeRGB videoToSequence videoDecodeCaps; do
  echo "=== Building $sample ==="
  mkdir -p "$PROJECT_ROOT/samples/${sample}/build"
  cmake -S "$PROJECT_ROOT/samples/${sample}" -B "$PROJECT_ROOT/samples/${sample}/build" 2>&1 | tail -2
  cmake --build "$PROJECT_ROOT/samples/${sample}/build" --target clean "${CONFIG_ARGS[@]}" 2>&1 | tail -2
  cmake --build "$PROJECT_ROOT/samples/${sample}/build" --parallel "$JOBS" "${CONFIG_ARGS[@]}" 2>&1 | tail -2
done

echo "=== All sample apps built ==="
