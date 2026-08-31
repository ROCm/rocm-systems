#!/bin/bash
# Build every experiment. The only build entry point.
#
#   ./build.sh                 build all
#   ./build.sh isolated_copy   build one
set -u
cd "$(dirname "$0")"

HIPCC=${HIPCC:-/opt/rocm/bin/hipcc}
ARCH=${ARCH:-gfx1250}
FLAGS="--offload-arch=$ARCH -O3 -std=c++17"

EXPERIMENTS=(
  isolated_copy       # headline table: 9 variants, cold streaming copy
  size_curve          # where in the size range the hint does anything
  cache_capacity      # how big GL2 actually is, and what survives a dispatch
  flush_sensitivity   # is the flush big enough for "cold" to mean cold
  concurrency         # copy against a co-running cache-resident victim
  adversarial         # deliberate attempts to make the hint lose
  small_copy          # dispatch-overhead-dominated sizes
  residency           # allocation type vs cross-dispatch GL2 retention
  e2e_memcpy          # hipMemcpyAsync through a patched CLR (needs clr_build.sh)
)

mkdir -p build
targets=("$@")
[ ${#targets[@]} -eq 0 ] && targets=("${EXPERIMENTS[@]}")

fail=0
for e in "${targets[@]}"; do
  src="src/experiments/${e}.hip"
  if [ ! -f "$src" ]; then
    printf '  %-20s SKIP (no %s)\n' "$e" "$src"
    continue
  fi
  if $HIPCC $FLAGS "$src" -o "build/$e" 2>"build/${e}.buildlog"; then
    printf '  %-20s ok\n' "$e"
  else
    printf '  %-20s FAILED\n' "$e"
    sed -n '1,25p' "build/${e}.buildlog"
    fail=1
  fi
done
exit $fail
