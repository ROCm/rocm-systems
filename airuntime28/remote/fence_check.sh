#!/bin/bash
# Is the fence scope what removes cross-dispatch cache residency?
#
# Merges what were fence_scope_check.sh and fence_root_cause.sh - two scripts
# asking one question in two halves, with the second half re-running the first.
#
# The question. cache_capacity shows nothing survives a dispatch boundary. The
# obvious explanation is the acquire fence at dispatch start issuing a GL2
# invalidate. Per the fence table in ROCM-11953, AGENT-scope acquire does not
# issue GL2_INV, only SYSTEM-scope does - so if these dispatches carry AGENT
# scope, the fence is not the cause and the mechanism is something else.
#
# The method. Do not reason about which branch the runtime takes; read the
# decoded packet headers CLR logs at AMD_LOG_LEVEL=4, then force SYSTEM scope
# everywhere with AMD_OPT_FLUSH=0 and see whether residency changes. If the
# latency scan is unchanged under the strongest fence available, the fence is
# not what removes residency.
#
# Needs the patched CLR build (clr_build.sh) because the logging and the
# AMD_OPT_FLUSH path are runtime-side.
set -uo pipefail
cd "$(dirname "$0")"
export LD_LIBRARY_PATH=~/airuntime28-clr-install/lib:/opt/rocm/lib

STRIP='rj warn|Resource leak|LoadLib|Secondary CUID|^ROW'

echo "=== 1. what ISA version does the runtime think this is ==="
# PR #966's gfx12 predicate is major==12 && minor==0 && stepping in {0,1}.
# gfx1250 is major 12 / minor 5 / stepping 0, which does NOT match - so that
# predicate cannot be the explanation, whatever else is.
/opt/rocm/bin/rocminfo 2>/dev/null | grep -m1 -A2 'amdgcn-amd-amdhsa--gfx1250' || true

echo
echo "=== 2. fence scopes actually submitted (1=AGENT, 2=SYSTEM) ==="
AMD_LOG_LEVEL=4 ./build/e2e_memcpy --iters 3 --warmup 1 > /tmp/fence_default.log 2>&1 || true
grep -oE 'type=[0-9]+, barrier=[0-9]+, acquire=[0-9]+, release=[0-9]+' /tmp/fence_default.log \
  | sort | uniq -c | sort -rn | head -12

echo
echo "--- scopes on dispatches of the blit copy kernel specifically ---"
grep -B2 -A2 '__amd_rocclr_copyBuffer' /tmp/fence_default.log \
  | grep -oE 'acquire=[0-9]+, release=[0-9]+' | sort | uniq -c | head

echo
echo "=== 3. does AMD_OPT_FLUSH=0 really change the packets ==="
AMD_OPT_FLUSH=0 AMD_LOG_LEVEL=4 ./build/e2e_memcpy --iters 3 --warmup 1 > /tmp/fence_system.log 2>&1 || true
grep -oE 'acquire=[0-9]+, release=[0-9]+' /tmp/fence_system.log | sort | uniq -c | sort -rn | head -5

echo
echo "=== 4. residency scan under the default fence scope ==="
./build/cache_capacity --residency-only --iters 10 --warmup 3 2>&1 \
  | grep -vE "$STRIP" | grep -E 'footprint|KiB|MiB|ratio'

echo
echo "=== 5. residency scan with SYSTEM scope forced everywhere ==="
AMD_OPT_FLUSH=0 ./build/cache_capacity --residency-only --iters 10 --warmup 3 2>&1 \
  | grep -vE "$STRIP" | grep -E 'footprint|KiB|MiB|ratio'

echo
echo "Read: if 4 and 5 agree row for row, forcing the strongest fence does not"
echo "restore residency, so the acquire fence is not what removes it."
echo "FENCE_CHECK_DONE"
