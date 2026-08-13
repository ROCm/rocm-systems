#!/usr/bin/env bash
# MIT License
#
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
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Builds the reproducers that need a compiler. Nothing here is wired into CTest: these are
# run by hand while investigating a specific bug.
#
#   ./build.sh --sdk-root /opt/rocm            # installed SDK
#   ./build.sh --sdk-root /path/to/build/stage # or a staged build tree
#
# The copy-failure shim needs no ROCm at all and is always built.

set -eu

SDK_ROOT="${SDK_ROOT:-/opt/rocm}"
HIPCC="${HIPCC:-}"
OUTDIR="${OUTDIR:-.}"

while [ $# -gt 0 ]; do
    case "$1" in
        --sdk-root) SDK_ROOT="$2"; shift 2 ;;
        --hipcc)    HIPCC="$2"; shift 2 ;;
        --outdir)   OUTDIR="$2"; shift 2 ;;
        -h|--help)  sed -n '24,30p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

mkdir -p "$OUTDIR"
HERE="$(cd "$(dirname "$0")" && pwd)"

echo "=== R5 copy-failure shim (no ROCm required) ==="
gcc -shared -fPIC -O2 -Wall -Wextra -Wno-unused-parameter \
    -o "$OUTDIR/libcopyfail.so" "$HERE/r5_copy_failure_shim.c" -ldl
echo "  $OUTDIR/libcopyfail.so"

if [ -z "$HIPCC" ]; then
    HIPCC="$(command -v hipcc || true)"
    [ -z "$HIPCC" ] && [ -x "$SDK_ROOT/bin/hipcc" ] && HIPCC="$SDK_ROOT/bin/hipcc"
fi
if [ -z "$HIPCC" ] || [ ! -x "$HIPCC" ]; then
    echo
    echo "SKIP: hipcc not found, so the HIP reproducers were not built."
    echo "      Pass --hipcc <path> or --sdk-root <rocm prefix>."
    exit 77
fi
echo
echo "using hipcc: $HIPCC"
echo "using sdk:   $SDK_ROOT"

INC="-I$SDK_ROOT/include"
LIB="-L$SDK_ROOT/lib -lrocprofiler-sdk -Wl,-rpath,$SDK_ROOT/lib"

echo
echo "=== tool library shared by R3, R4 and R6 ==="
"$HIPCC" -O2 -std=c++17 -fPIC -shared $INC \
    -o "$OUTDIR/librepro_client.so" "$HERE/repro_client.cpp" $LIB
echo "  $OUTDIR/librepro_client.so"

echo
echo "=== R3 concurrent hipFree during a replay window ==="
"$HIPCC" -O2 -std=c++17 -o "$OUTDIR/r3_free_during_replay" "$HERE/r3_free_during_replay.cpp"
echo "  $OUTDIR/r3_free_during_replay"

echo
echo "=== R6 module variable size cap (two configurations) ==="
small=$((300 * 1024 * 1024 / 4))
big=$((1300 * 1024 * 1024 / 4))
"$HIPCC" -O2 -std=c++17 -DKR_BIG_ELEMS=$small \
    -o "$OUTDIR/r6_under_cap" "$HERE/r6_large_module_variable.cpp"
echo "  $OUTDIR/r6_under_cap   (300 MiB device variable, control)"
"$HIPCC" -O2 -std=c++17 -DKR_BIG_ELEMS=$big \
    -o "$OUTDIR/r6_over_cap" "$HERE/r6_large_module_variable.cpp" || {
    echo "  r6_over_cap failed to build -- a 1.3 GiB __device__ array may exceed a limit here."
    echo "  That is itself worth noting, but it is not the bug under test."
}
[ -x "$OUTDIR/r6_over_cap" ] && echo "  $OUTDIR/r6_over_cap    (1300 MiB device variable, subject)"

cat <<NOTES

built. next:

  # R3
  LD_PRELOAD=$OUTDIR/librepro_client.so KR_REPRO_PASSES=8 $OUTDIR/r3_free_during_replay

  # R4 and R7
  ./r4_r7_bounded_and_csv.sh --app <any hip app> --client $OUTDIR/librepro_client.so

  # R6
  LD_PRELOAD=$OUTDIR/librepro_client.so KR_REPRO_PASSES=4 $OUTDIR/r6_under_cap
  LD_PRELOAD=$OUTDIR/librepro_client.so KR_REPRO_PASSES=4 $OUTDIR/r6_over_cap

  # R8 needs nothing
  ./r8_tolerance_asymmetry.py
NOTES
