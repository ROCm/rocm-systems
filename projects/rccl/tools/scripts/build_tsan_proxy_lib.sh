#!/usr/bin/env bash
# Rebuild librccl.so with ThreadSanitizer instrumentation on src/proxy.cc only.
#
# The main-thread <-> proxy-thread free-list race lives entirely inside
# proxy.cc, so instrumenting that single translation unit (it is already built
# --offload-host-only, i.e. pure host code) is enough for TSAN to judge the real
# RCCL code, without paying for a whole-library sanitizer build. The sanitizer
# runtime itself is not linked in here - see the relink step below - the caller
# preloads it into the process instead.
#
# Usage: build_tsan_proxy_lib.sh <build_dir> <out_dir> [source]
# The optional source replaces the hipified proxy.cc for this compile, which is
# how the caller feeds in a shimmed copy without touching the source tree.
# Must run where the build's compiler and ROCm are available. Nothing is assumed
# about their paths: the compiler is taken from the build directory itself.
set -euo pipefail

BD="${1:?usage: $0 <build_dir> <out_dir> [source]}"
OUT="${2:?usage: $0 <build_dir> <out_dir> [source]}"
D="$BD/src/CMakeFiles/rccl.dir"
SRC="${3:-$BD/hipify/src/proxy.cc}"
OBJ="$D/__/hipify/src/proxy.cc.o"

[ -f "$D/flags.make" ] || { echo "missing $D/flags.make"; exit 1; }
[ -f "$D/link.txt" ]   || { echo "missing $D/link.txt"; exit 1; }
[ -f "$SRC" ]          || { echo "missing source to compile: $SRC"; exit 1; }

CXX_DEFINES=$(sed -n 's/^CXX_DEFINES = //p'  "$D/flags.make")
CXX_INCLUDES=$(sed -n 's/^CXX_INCLUDES = //p' "$D/flags.make")
CXX_FLAGS=$(sed -n 's/^CXX_FLAGS = //p'      "$D/flags.make")

# Use the compiler the build itself uses, so this keeps working under a different
# ROCm prefix or a compiler wrapper instead of assuming one fixed path.
CXX=$(sed -n 's/^CMAKE_CXX_COMPILER:[A-Z]*=//p' "$BD/CMakeCache.txt")
[ -x "$CXX" ] || { echo "cannot run the build's C++ compiler (CMAKE_CXX_COMPILER='$CXX')"; exit 1; }

# CONSUMER_FN is the function the caller attributes reports to. Inlining it would
# leave the sanitizer naming its caller instead, which reads as "no race" on code
# that is racy, so keep it out of line and check below that it stayed there.
CONSUMER_FN="${CONSUMER_FN:-ncclLocalOpAppend}"

echo "[tsan] recompiling $(basename "$SRC") with -fsanitize=thread ($CXX)"
# shellcheck disable=SC2086
"$CXX" $CXX_DEFINES $CXX_INCLUDES $CXX_FLAGS \
  -fsanitize=thread -fno-omit-frame-pointer -fno-inline -g \
  -o "$OBJ" -c "$SRC"

# If the instrumentation silently did not happen the run would report no races
# at all, which reads like a clean result, so check the object for it.
OBJREFS=$(nm -u "$OBJ" | grep -c "__tsan_" || true)
[ "$OBJREFS" -gt 0 ] || { echo "$OBJ has no __tsan_ references - not instrumented"; exit 1; }

# Same reasoning for the frame the caller keys on: if it is gone, whether inlined
# away or renamed upstream, reports would be attributed elsewhere and go uncounted.
FRAMES=$(nm -C "$OBJ" | grep -c "$CONSUMER_FN" || true)
[ "$FRAMES" -gt 0 ] || { echo "$OBJ has no '$CONSUMER_FN' symbol, so reports could not be attributed to it"; exit 1; }

echo "[tsan] relinking librccl.so.1.0 with the instrumented proxy.cc"
# The link command is used exactly as the build wrote it. -fsanitize=thread is
# deliberately not injected: clang does not link the sanitizer runtime into a
# shared library, it expects the process to bring it, which is why the runner
# preloads it. Injecting the flag here would only look like it did something.
# Read link.txt before the subshell: its own paths are relative to $BD/src.
LINKCMD=$(cat "$D/link.txt")
( cd "$BD/src" && eval "$LINKCMD" )

mkdir -p "$OUT"
# Plain cp: only the contents matter, and preserving attributes fails outright on
# filesystems that do not support them.
cp -f "$BD/librccl.so.1.0" "$OUT/"
( cd "$OUT" && ln -sf librccl.so.1.0 librccl.so.1 && ln -sf librccl.so.1.0 librccl.so )
# The check that actually matters: the library handed to the runner has to carry
# the instrumentation. If a relink silently does nothing, the run reports no
# races and that reads like a clean result.
REFS=$(nm -u "$OUT/librccl.so.1.0" | grep -c "__tsan_" || true)
[ "$REFS" -gt 0 ] || { echo "$OUT/librccl.so.1.0 carries no __tsan_ references - the relink did not pick up the instrumented object"; exit 1; }
echo "[tsan] wrote $OUT/librccl.so.1.0 ($REFS __tsan_ references, runtime comes from LD_PRELOAD)"
