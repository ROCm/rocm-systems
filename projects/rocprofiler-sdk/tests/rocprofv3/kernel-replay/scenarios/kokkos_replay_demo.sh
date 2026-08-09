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
# Kokkos application under kernel replay.
#
# Why Kokkos is a distinct test from the microbenchmark: a Kokkos app launches many small
# kernels with mangled functor names, may use several execution space instances mapping to
# several HIP streams, and issues its own fences. That exercises three things the CI
# microbenchmark does not:
#
#   1. per-dispatch opt-out at scale -- one hot kernel replayed, hundreds of others not,
#      which is the realistic customer shape and the case that pays the reader-lock tax;
#   2. the agent-wide sibling-queue drain, because other streams have work in flight when
#      a replay window opens;
#   3. Kokkos fences interleaving with the replay window's blocking waits, which is where
#      a deadlock would show up.
#
# Loading the Kokkos Tools connector at the same time checks that replay composes with an
# external profiling interface rather than fighting it.
#
#   ./kokkos_replay_demo.sh --app ./my_kokkos_app --kernel "SomeFunctor"
#
# Any Kokkos executable works. Kokkos' own benchmarks or a mini-app such as ExaMiniMD are
# reasonable choices; a single parallel_for over a large View is enough to be meaningful.

set -u

ROCPROFV3="${ROCPROFV3:-$(command -v rocprofv3 || echo rocprofv3)}"
APP=""
APP_ARGS=""
KERNEL=""
KOKKOS_TOOL="${KOKKOS_TOOLS_LIBS:-}"
OUT="${OUT:-kokkos-replay-out}"

while [ $# -gt 0 ]; do
    case "$1" in
        --rocprofv3)  ROCPROFV3="$2"; shift 2 ;;
        --app)        APP="$2"; shift 2 ;;
        --args)       APP_ARGS="$2"; shift 2 ;;
        --kernel)     KERNEL="$2"; shift 2 ;;
        --kokkos-tool) KOKKOS_TOOL="$2"; shift 2 ;;
        --out)        OUT="$2"; shift 2 ;;
        -h|--help)    sed -n '25,45p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
done

if ! command -v "$ROCPROFV3" >/dev/null 2>&1 && [ ! -x "$ROCPROFV3" ]; then
    echo "SKIP: rocprofv3 not found ($ROCPROFV3)"; exit 77
fi
if [ -z "$APP" ] || [ ! -x "$APP" ]; then
    echo "SKIP: --app must point at a Kokkos executable (got '${APP}')"
    echo "      Build any Kokkos app with the HIP backend and pass it here."
    exit 77
fi

mkdir -p "$OUT"
G1="SQ_WAVES SQ_INSTS_VALU GRBM_COUNT"
G2="SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE"
G3="SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SALU"
RC=0

echo "=== K0: baseline, no profiling ==="
echo "  Establishes the application's own wall time, which the cost model compares against."
/usr/bin/time -f "  wall %e s   peak RSS %M KiB" "$APP" $APP_ARGS >"$OUT/k0.log" 2>"$OUT/k0.time" \
    && cat "$OUT/k0.time" || { echo "  FAILED"; tail -5 "$OUT/k0.log"; RC=1; }

echo
echo "=== K1: three counter groups, whole application replayed ==="
echo "  Every dispatch replayed three times. Upper bound on replay cost for this app."
# shellcheck disable=SC2086
if "$ROCPROFV3" --pmc $G1 --pmc $G2 --pmc $G3 --kernel-replay-beta-enabled \
        --output-format json -d "$OUT/k1" -o out -- "$APP" $APP_ARGS >"$OUT/k1.log" 2>&1; then
    echo "  ok"
else
    echo "  FAILED (rc=$?):"; tail -10 "$OUT/k1.log" | sed 's/^/      /'; RC=1
fi

echo
echo "=== K2: one hot kernel replayed, the rest opted out ==="
if [ -z "$KERNEL" ]; then
    echo "  SKIP: pass --kernel <regex> to select the hot kernel."
    echo "        Kokkos functor names are mangled; take one from the K1 output above."
else
    echo "  This is the realistic shape: replay narrowed to '$KERNEL', everything else"
    echo "  single-pass. Compare against K1 to see what the opt-out path saves."
    # shellcheck disable=SC2086
    if "$ROCPROFV3" --pmc $G1 --pmc $G2 --pmc $G3 --kernel-replay-beta-enabled \
            --kernel-include-regex "$KERNEL" --output-format json \
            -d "$OUT/k2" -o out -- "$APP" $APP_ARGS >"$OUT/k2.log" 2>&1; then
        echo "  ok"
    else
        echo "  FAILED (rc=$?):"; tail -10 "$OUT/k2.log" | sed 's/^/      /'; RC=1
    fi
fi

echo
echo "=== K3: replay with the Kokkos Tools connector loaded ==="
if [ -z "$KOKKOS_TOOL" ]; then
    echo "  SKIP: set KOKKOS_TOOLS_LIBS or pass --kokkos-tool <libkp_*.so>."
    echo "        Checks that replay coexists with an external profiling interface;"
    echo "        note the connector sees one logical launch while replay executes N."
else
    # shellcheck disable=SC2086
    if KOKKOS_TOOLS_LIBS="$KOKKOS_TOOL" "$ROCPROFV3" --pmc $G1 --pmc $G2 \
            --kernel-replay-beta-enabled --output-format json \
            -d "$OUT/k3" -o out -- "$APP" $APP_ARGS >"$OUT/k3.log" 2>&1; then
        echo "  ok -- confirm the connector's kernel count matches the application's own"
        echo "        launch count, not the replayed pass count"
    else
        echo "  FAILED (rc=$?):"; tail -10 "$OUT/k3.log" | sed 's/^/      /'; RC=1
    fi
fi

echo
echo "=== what to check in the output ==="
cat <<'NOTES'
  * Shared counters (SQ_WAVES, SQ_INSTS_VALU) constant across a dispatch's passes.
    Kokkos Views are hipMalloc-backed, so the tracker should cover them; variation here
    means something in the working set escaped the snapshot.
  * The application's own results unchanged versus K0. Kokkos mini-apps usually print a
    convergence or energy value -- it must match the unprofiled run.
  * No hang. A stall in K1 or K2 with several streams active points at the sibling-queue
    drain or a Kokkos fence interacting with the replay window's blocking waits.
  * Dispatch count. The application should observe its own number of launches; only the
    counter records multiply by the pass count.
NOTES
exit "$RC"
