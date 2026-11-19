#!/usr/bin/env bash
# Runs every reproducer build.sh built, first as it is and then with each of
# its fixes, and prints one line for each run.
#
#   usage: run-all.sh DIRECTORY [BIN_DIR]
#
# DIRECTORY is where the temporary files go; fastpath needs an AIS-capable
# file system there. BIN_DIR is where build.sh put the programs (default:
# ./bin). Run this with nothing else using the GPU: another process's work
# changes what the GPU's caches hold, and can turn a failure into a pass.

here=$(cd "$(dirname "$0")" && pwd)
directory=${1:?usage: run-all.sh DIRECTORY [BIN_DIR]}
bin=${2:-$here/bin}
# ROCm's libraries; the programs find the libhipfile.so they were built against first
export LD_LIBRARY_PATH=${ROCM_PATH:-/opt/rocm}/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}

for program in "$bin"/[0-9][0-9]-*; do
    [ -x "$program" ] || continue
    # It hung the amdgpu driver on RX 9070 (gfx1201); run it only when asked to
    if [[ $program == *-fallback-read-during-kernel && -z ${REPRO_MAY_HANG_DRIVER:-} ]]; then
        printf '%-52s %s\n' "$(basename "$program")" "skipped: set REPRO_MAY_HANG_DRIVER=1 to run it"
        continue
    fi
    # The usage message lists the fixes: "  FIX is one of: none, a, b"
    fixes=$("$program" 2>&1 | sed -n 's/^ *FIX is one of: //p' | tr -d ',')
    if [ -z "$fixes" ]; then
        printf '%-52s %s\n' "$(basename "$program")" "does not start: $("$program" 2>&1 | head -n 1)"
        continue
    fi
    for fix in $fixes; do
        result=$(timeout 300 "$program" "$directory" "$fix" 2>&1 | tail -n 1)
        printf '%-52s %s\n' "$(basename "$program")" "$result"
    done
done
