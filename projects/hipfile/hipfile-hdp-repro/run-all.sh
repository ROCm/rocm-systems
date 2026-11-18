#!/bin/bash
# Runs every hipFile data-visibility test and probe from the HDP / L2 investigation on one node:
# Kurt's hipfileread-cpu-visibility, John's hipfileread-gpu-visibility and hipfilewrite-visibility, their ctest entries, the
# HIP-only BAR repro, and the subagents' probes (a1 barprobe + instrumented Kurt test, a2
# stale-probe, c f1probe/f2probe). See README.md for what each section does and what touches
# the BAR or the HDP register.
#
#   ./run-all.sh --hipfile-src <checkout>/projects/hipfile --dir <AIS-capable dir> [options]
#
# Options:
#   --hipfile-src DIR   projects/hipfile of a checkout of users/john00003/hipfile-read-gpu-visibility-test
#   --dir DIR           directory on an AIS-capable (ext4 data=ordered / xfs) NVMe mount for test files
#   --build-dir DIR     hipFile build directory (default: <hipfile-src>/build-visibility)
#   --build             configure and build hipFile even if the build directory exists
#   --sections LIST     comma-separated; default: health,ctest,kurt,gpu,write,aqllog,hipbar,a1,a2,c
#                       (hdpprobe is added by --allow-hdp-flush)
#   --allow-hdp-flush   also run configs that write the HDP flush register (write only, never read
#                       back) and the hdpprobe section. Off by default.
#   --gpu N             run on GPU N only (sets ROCR_VISIBLE_DEVICES)
#   --quick             fewer iterations and repetitions, for a smoke run
#   --out DIR           results directory (default: ./results-<host>-<timestamp>)
#   --dry-run           print the commands instead of running them
#
# Environment: ROCM_PATH (default /opt/rocm), HIPCXX (default amdclang++), EXTRA_FLAGS (added to
# every probe compile, e.g. --gcc-install-dir=/usr/lib/gcc/x86_64-linux-gnu/13).

ROCM=${ROCM_PATH:-/opt/rocm}
HIPCXX=${HIPCXX:-amdclang++}
EXTRA_FLAGS=${EXTRA_FLAGS:-}
HERE=$(cd "$(dirname "$0")" && pwd)
SRC='' DIR='' BUILD='' DO_BUILD=0 SECTIONS='' ALLOW_HDP=0 QUICK=0 GPU='' OUT='' DRY=0

die() { echo "error: $*" >&2; exit 2; }
while [ $# -gt 0 ]; do
    case $1 in
        --hipfile-src) SRC=$(realpath "$2"); shift 2 ;;
        --dir) DIR=$(realpath "$2"); shift 2 ;;
        --build-dir) BUILD=$(realpath -m "$2"); shift 2 ;;
        --build) DO_BUILD=1; shift ;;
        --sections) SECTIONS=$2; shift 2 ;;
        --allow-hdp-flush) ALLOW_HDP=1; shift ;;
        --gpu) GPU=$2; shift 2 ;;
        --quick) QUICK=1; shift ;;
        --out) OUT=$(realpath -m "$2"); shift 2 ;;
        --dry-run) DRY=1; shift ;;
        -h | --help) sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
[ -n "$SRC" ] && [ -f "$SRC/test/amd_detail/hipfileread-gpu-visibility.cpp" ] ||
    die "--hipfile-src must point at projects/hipfile of a checkout with hipfileread-gpu-visibility"
[ -n "$DIR" ] && [ -d "$DIR" ] && [ -w "$DIR" ] || die "--dir must be an existing writable directory"
BUILD=${BUILD:-$SRC/build-visibility}
SECTIONS=${SECTIONS:-health,ctest,kurt,gpu,write,aqllog,hipbar,a1,a2,c}
[ "$ALLOW_HDP" = 1 ] && [[ ,$SECTIONS, != *,hdpprobe,* ]] && SECTIONS+=,hdpprobe
[ -n "$GPU" ] && export ROCR_VISIBLE_DEVICES=$GPU
OUT=${OUT:-$PWD/results-$(hostname -s)-$(date +%Y%m%d-%H%M%S)}
mkdir -p "$OUT/bin" "$OUT/logs" || die "cannot create $OUT"
SUMMARY=$OUT/summary.txt
exec > >(tee -a "$OUT/run.log") 2>&1

if [ "$QUICK" = 1 ]; then REPS=1 KURT_ITERS=200 GPU_ITERS=50 BIG_ITERS=5 QUICK_ITERS=ITERS=100
else REPS=3 KURT_ITERS=2000 GPU_ITERS=200 BIG_ITERS=50 QUICK_ITERS=; fi
if [ "$QUICK" = 1 ]; then WRITE_ITERS=100 SLEEP_ITERS=20; else WRITE_ITERS=1000 SLEEP_ITERS=100; fi

note() { echo "$*" | tee -a "$SUMMARY"; }
section() { note ""; note "######## $1 ########"; }
has_section() { [[ ,$SECTIONS, == *,$1,* ]]; }
# Lines that touch the HDP flush register are skipped unless --allow-hdp-flush
# (labels don't count, and HSA_ENABLE_SDMA_HDP_FLUSH=0 only stops ROCr adding its own SDMA flush packet)
hdp_line() { sed -E 's/LABEL=[^ ]*//g; s/HSA_ENABLE_SDMA_HDP_FLUSH=[0-9]+//g' <<<"$1" | grep -qiE 'hdp'; }
# run LOGNAME CMD...: runs with a timeout, appends output to logs/LOGNAME.log, prints the output
run() {
    local log=$OUT/logs/$1.log
    shift
    echo "\$ $*" >>"$log"
    if [ "$DRY" = 1 ]; then echo "DRY: $*"; return 0; fi
    timeout 900 "$@" 2>&1 | tee -a "$log"
    return "${PIPESTATUS[0]}"
}
gpu_busy_check() {
    if timeout 20 "$ROCM/bin/rocm-smi" --showpids 2>&1 | grep -q 'KFD process information'; then
        note "WARNING: another process is using the GPU before section $1; results may be disturbed:"
        timeout 20 "$ROCM/bin/rocm-smi" --showpids 2>&1 | grep -E '^ *[0-9]+ ' | tee -a "$SUMMARY"
    fi
}

exec 9>"/tmp/hipfile-visibility-$USER.lock"
flock -n 9 || die "another run-all.sh holds /tmp/hipfile-visibility-$USER.lock"

note "run-all.sh on $(hostname) at $(date -Is)"
note "hipfile-src=$SRC ($(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null) $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null))"
note "dir=$DIR build=$BUILD sections=$SECTIONS allow-hdp-flush=$ALLOW_HDP quick=$QUICK gpu=${GPU:-all} out=$OUT"

# ---- build hipFile -------------------------------------------------------------------------
if [ "$DO_BUILD" = 1 ] || [ ! -x "$BUILD/test/amd_detail/hipfileread-gpu-visibility" ] ||
    [ ! -x "$BUILD/test/amd_detail/hipfilewrite-visibility" ]; then
    section "build hipFile"
    cmake_flags=(-DCMAKE_CXX_COMPILER="$HIPCXX" -DCMAKE_BUILD_TYPE=Debug -DHIPFILE_CAPABLE_DIR="$DIR")
    [ -n "$EXTRA_FLAGS" ] && cmake_flags+=(-DCMAKE_CXX_FLAGS="$EXTRA_FLAGS" -DCMAKE_HIP_FLAGS="$EXTRA_FLAGS")
    run build cmake -S "$SRC" -B "$BUILD" "${cmake_flags[@]}" >/dev/null ||
        die "cmake configure failed; see $OUT/logs/build.log"
    LD_LIBRARY_PATH=$BUILD/src/amd_detail:$ROCM/lib run build cmake --build "$BUILD" --parallel \
        --target hipfile hipfileread-cpu-visibility hipfileread-gpu-visibility hipfilewrite-visibility \
        hipfile-fastpath-gate >/dev/null ||
        die "hipFile build failed; see $OUT/logs/build.log"
fi
# /opt/rocm/lib may hold an older libhipfile, and LD_LIBRARY_PATH beats RUNPATH: build dir first
export LD_LIBRARY_PATH=$BUILD/src/amd_detail:$ROCM/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}
T=$BUILD/test/amd_detail
KURT=$T/hipfileread-cpu-visibility
GVIS=$T/hipfileread-gpu-visibility
WVIS=$T/hipfilewrite-visibility
ldd "$GVIS" | grep -q "$BUILD/src/amd_detail/libhipfile" || die "$GVIS does not load $BUILD's libhipfile"
CAPDIR=$(grep -oP '^HIPFILE_CAPABLE_DIR:\w+=\K.*' "$BUILD/CMakeCache.txt")
[ "$CAPDIR" = "$DIR" ] || note "NOTE: ctest uses HIPFILE_CAPABLE_DIR=$CAPDIR from the existing build, not --dir; pass --build to reconfigure"

ARCH=$("$ROCM/llvm/bin/amdgpu-arch" 2>/dev/null | head -1)
[ -n "$ARCH" ] || ARCH=$("$ROCM/bin/rocminfo" 2>/dev/null | grep -m1 -oE 'gfx[0-9a-f]+')
note "GPU arch: ${ARCH:-unknown}; CPUs: $(nproc)"

# Fastpath is usable for --dir if a fastpath-only read succeeds
FASTPATH=0
if [ "$DRY" = 1 ]; then
    FASTPATH=1
elif HIPFILE_ALLOW_COMPAT_MODE=false timeout 120 "$GVIS" --directory "$DIR" --iterations 1 --kernel after >/dev/null 2>&1; then
    FASTPATH=1
fi
note "fastpath available for $DIR: $([ $FASTPATH = 1 ] && echo yes || echo NO - fastpath runs are skipped)"
BACKENDS=(fallback)
[ "$FASTPATH" = 1 ] && BACKENDS=(fastpath fallback)
backend_env() { [ "$1" = fastpath ] && echo HIPFILE_ALLOW_COMPAT_MODE=false || echo HIPFILE_FORCE_COMPAT_MODE=true; }

# ---- probe builds ----------------------------------------------------------------------------
# hip_build OUT SRC [flags...]: compile a HIP probe for this node's GPU
hip_build() {
    local out=$1 src=$2
    shift 2
    [ -x "$OUT/bin/$out" ] && return 0
    run probe-build "$HIPCXX" -x hip --offload-arch="$ARCH" -O2 -std=c++20 $EXTRA_FLAGS "$src" -o "$OUT/bin/$out" \
        -L"$ROCM/lib" -Wl,-rpath,"$ROCM/lib" "$@" >/dev/null || { note "BUILD FAILED: $out (see logs/probe-build.log)"; return 1; }
}
HIPFILE_CFLAGS=(-DAIS_TESTING -D__HIP_PLATFORM_AMD__ -I"$SRC/shared" -isystem "$SRC/src/amd_detail" -isystem "$SRC/include"
                -isystem "$SRC/test/common" -isystem "$SRC/third-party" -isystem "$ROCM/include")
# shellcheck disable=SC2054  # the comma in -Wl,-rpath is part of the flag
HIPFILE_LIBS=(-L"$BUILD/src/amd_detail" -Wl,-rpath,"$BUILD/src/amd_detail" -lhipfile)

# ---- health ----------------------------------------------------------------------------------
health() {
    bash "$HERE/health.sh" >"$OUT/logs/health-$1.log" 2>&1
    note "health $1: $(grep -cE '(correctable|nonfatal|fatal)=[1-9]' "$OUT/logs/health-$1.log") devices with nonzero AER," \
         "$(awk '/D state/{f=1;next} /\(end\)/{f=0} f' "$OUT/logs/health-$1.log" | grep -c .) D-state processes" \
         "(logs/health-$1.log)"
}
health before

# ---- ctest: Kurt's and John's registered tests --------------------------------------------
if has_section ctest; then
    section "ctest (hipfileread_cpu_visibility_*, hipfileread_gpu_visibility_* and hipfilewrite_visibility_*)"
    gpu_busy_check ctest
    run ctest ctest --test-dir "$BUILD" -R 'hipfileread_(cpu|gpu)_visibility|hipfilewrite_visibility' --output-on-failure |
        grep -E 'Test +#|tests passed' | tee -a "$SUMMARY"
fi

# ---- kurt: hipfileread-cpu-visibility matrix (CPU reads VRAM through the BAR) ---------------
kurt_verdict() {
    if grep -q 'DONE!' <<<"$1"; then echo "PASS"
    elif grep -q 'VRAM is not CPU-accessible' <<<"$1"; then echo "SKIP (VRAM not CPU-accessible)"
    elif grep -q 'passed before the failure' <<<"$1"; then
        echo "FAIL: $(grep -m1 -oE '[0-9]+ iterations passed before the failure' <<<"$1"); $(grep -m1 'Iteration' <<<"$1" | cut -c1-90)"
    else echo "ERROR: $(head -2 <<<"$1" | tr '\n' ' ' | cut -c1-120)"; fi
}
if has_section kurt; then
    section "kurt: hipfileread-cpu-visibility, $KURT_ITERS iterations, $REPS reps"
    gpu_busy_check kurt
    for be in "${BACKENDS[@]}"; do
        for cpus in $(printf '%s\n' 2 8 24 "$(nproc)" | sort -nu); do
            [ "$cpus" -le "$(nproc)" ] || continue
            for size in 64K 1M 4M; do
                for ((rep = 1; rep <= REPS; rep++)); do
                    out=$(run kurt env "$(backend_env "$be")" "$KURT" --directory "$DIR" --iterations "$KURT_ITERS" \
                          --cpus "$cpus" --buffer-size "$size")
                    note "$(printf '%-8s cpus=%-3s buf=%-3s rep %s: %s' "$be" "$cpus" "$size" "$rep" "$(kurt_verdict "$out")")"
                done
            done
        done
    done
fi

# ---- gpu: hipfileread-gpu-visibility matrix (GPU kernels read VRAM) --------------------------
gvis_verdict() {
    if grep -q 'passed in' <<<"$1"; then echo "PASS"
    elif grep -q 'stale and' <<<"$1"; then
        echo "FAIL at $(grep -m1 -oE 'Iteration [0-9]+' <<<"$1"): $(grep -c 'stale and' <<<"$1") workgroups," \
             "$(grep 'stale and' <<<"$1" | awk '{s+=$5} END{print s}') stale words"
    else echo "ERROR: $(grep -m1 -vE '^PROBE' <<<"$1" | cut -c1-120)"; fi
}
GVIS_CONFIGS=(
    "--kernel after"
    "--kernel after --prime --prime-wait sync"
    "--kernel after --prime --prime-wait flag"
    "--kernel after --prime --prime-wait sleep"
    "--kernel after --read async"
    "--kernel after --prime --prime-wait sync --read async"
    "--kernel after --prime --prime-wait flag --read async"
    "--kernel after --prime --prime-wait sleep --read async"
    "--kernel during"
    "--kernel during --prime"
)
GVIS_BIG_CONFIGS=(
    "--kernel after --prime --prime-wait sync"
    "--kernel after --prime --prime-wait flag"
    "--kernel during --prime"
)
if has_section gpu; then
    section "gpu: hipfileread-gpu-visibility, $GPU_ITERS iterations (1M) / $BIG_ITERS (256M), 2 reps"
    gpu_busy_check gpu
    for be in "${BACKENDS[@]}"; do
        for cfg in "${GVIS_CONFIGS[@]}"; do
            for rep in 1 2; do
                out=$(run gpu env "$(backend_env "$be")" "$GVIS" --directory "$DIR" --iterations "$GPU_ITERS" $cfg)
                note "$(printf '%-8s %-56s 1M   rep %s: %s' "$be" "$cfg" "$rep" "$(gvis_verdict "$out")")"
            done
        done
        for cfg in "${GVIS_BIG_CONFIGS[@]}"; do
            out=$(run gpu env "$(backend_env "$be")" "$GVIS" --directory "$DIR" --iterations "$BIG_ITERS" --buffer-size 256M $cfg)
            note "$(printf '%-8s %-56s 256M rep 1: %s' "$be" "$cfg" "$(gvis_verdict "$out")")"
        done
    done
fi

# ---- write: hipfilewrite-visibility matrix (the drive reads VRAM through the HDP) ------------
wvis_verdict() {
    if grep -q 'passed in' <<<"$1"; then echo "PASS"
    elif grep -q 'the file differed' <<<"$1"; then
        echo "FAIL at $(grep -m1 -oE 'iteration [0-9]+' <<<"$1"):" \
             "$(grep -m1 -oE '[0-9]+ of [0-9]+ 64-bit words' <<<"$1") wrong," \
             "second write $(grep -q 'wrote the generation' <<<"$1" && echo right || echo 'still wrong')," \
             "buffer $(grep -q 'held the generation' <<<"$1" && echo right || echo wrong)"
    else echo "ERROR: $(grep -m1 -v '^$' <<<"$1" | cut -c1-120)"; fi
}
# The buffer is 1M unless a configuration says otherwise
WVIS_CONFIGS=(
    # A kernel that finished before hipFileWrite, and each way the host can learn that it did
    "--producer kernel --wait sync"
    "--producer kernel --wait sync --prime write"
    "--producer kernel --wait query"
    "--producer kernel --wait event"
    "--producer kernel --wait callback"
    "--producer kernel --write async"
    "--producer kernel --wait flag"
    "--producer kernel --wait sleep"
    "--producer kernel --wait sleep --memory uncached"
    # A kernel still running during hipFileWrite
    "--producer kernel --kernel during"
    "--producer kernel --kernel during --memory uncached"
    "--producer kernel --kernel during --memory coarse"
    "--producer kernel --kernel during --buffer-size 4K --background-writers 4"
    "--producer kernel --kernel during --memory uncached --buffer-size 4K --background-writers 4"
    # hipMemcpy from host memory: a kernel copies up to 16K, the SDMA engine copies more
    "--producer memcpy --buffer-size 4K"
    "--producer memcpy --buffer-size 16K"
    "--producer memcpy --buffer-size 64K"
    "--producer memcpy"
    "--producer memcpy --buffer-size 4K --prime write"
    "--producer memcpy --buffer-size 4K --background-writers 1"
    "--producer memcpy --buffer-size 4K --background-writers 4"
    "--producer memcpy --buffer-size 4K --background-writers 4 --memory uncached"
    "--producer memcpy --buffer-size 16K --background-writers 4"
    "--producer memcpy --buffer-size 64K --background-writers 8"
    "--producer memcpy --buffer-size 4K --threads 16"
    # Synchronous HIP calls that fill the buffer with a kernel
    "--producer memset --buffer-size 4K"
    "--producer memset"
    "--producer memset --memory uncached"
    "--producer memset --memory fine"
    "--producer copy --buffer-size 4K"
    "--producer copy"
    # hipFileRead, then hipFileWrite
    "--producer read --buffer-size 4K"
    "--producer read"
    "--producer read --prime write"
    "--producer read --buffer-size 4K --background-writers 4"
    "--producer read --buffer-size 4K --threads 16"
    "--producer read --prime store --wait sync --threads 4 --buffer-size 4M"
    "--producer read --prime store --wait sleep --threads 4 --buffer-size 4M"
    # A kernel the host synchronizes with, with the HDP's read path busy
    "--producer kernel --wait sync --buffer-size 4K --background-writers 4"
    "--producer kernel --wait sync --threads 4"
)
WVIS_BIG_CONFIGS=(
    "--producer memset --buffer-size 64M"
    "--producer copy --buffer-size 64M"
    "--producer read --buffer-size 16M"
)
# The AQL packets HIP submits show why a configuration fails: the fills that
# fail are dispatches with agent-scope fences and no barrier before the write
WVIS_AQL_CONFIGS=(
    "--producer kernel --wait sync"
    "--producer kernel --wait query"
    "--producer kernel --wait sleep"
    "--producer memset"
    "--producer memset --memory fine"
    "--producer copy --buffer-size 4K"
    "--producer copy"
    "--producer memcpy --buffer-size 4K"
)
if has_section write; then
    section "write: hipfilewrite-visibility, $WRITE_ITERS iterations ($SLEEP_ITERS with --wait sleep, $BIG_ITERS for 16M and 64M)"
    gpu_busy_check write
    for be in "${BACKENDS[@]}"; do
        for cfg in "${WVIS_CONFIGS[@]}"; do
            iters=$WRITE_ITERS
            [[ $cfg == *"--wait sleep"* ]] && iters=$SLEEP_ITERS
            read -ra cfg_args <<<"$cfg"
            out=$(run write env "$(backend_env "$be")" "$WVIS" --directory "$DIR" --iterations "$iters" \
                  "${cfg_args[@]}")
            note "$(printf '%-8s %-90s %s' "$be" "$cfg" "$(wvis_verdict "$out")")"
        done
        for cfg in "${WVIS_BIG_CONFIGS[@]}"; do
            read -ra cfg_args <<<"$cfg"
            out=$(run write env "$(backend_env "$be")" "$WVIS" --directory "$DIR" --iterations "$BIG_ITERS" \
                  "${cfg_args[@]}")
            note "$(printf '%-8s %-90s %s' "$be" "$cfg" "$(wvis_verdict "$out")")"
        done
    done
    note "AQL packet headers for 3 iterations (acquire and release fences: 0xb02 agent, 0x1502 and 0x1503 system):"
    be=${BACKENDS[0]}
    for cfg in "${WVIS_AQL_CONFIGS[@]}"; do
        read -ra cfg_args <<<"$cfg"
        out=$(run write-aql env AMD_LOG_LEVEL=5 AMD_LOG_MASK=8 "$(backend_env "$be")" "$WVIS" --directory "$DIR" \
              --iterations 3 "${cfg_args[@]}" 2>&1)
        seq=$(grep -oE '(Dispatch|Barrier-?AND|Barrier-?OR|Marker)[^,]*Header = 0x[0-9a-f]+' <<<"$out" |
              sed -E 's/ Header = /=/' | tr '\n' ' ')
        note "$be $cfg: $seq-> $(wvis_verdict "$out")"
    done
fi

# ---- aqllog: which AQL packets HIP submits around the read (explains sync vs async) ---------
if has_section aqllog; then
    section "aqllog: AQL packet headers for --prime-wait flag with --read sync vs async (acquire=1 agent, 2 system)"
    be=${BACKENDS[0]}
    for read in sync async; do
        out=$(run aqllog env AMD_LOG_LEVEL=5 AMD_LOG_MASK=8 "$(backend_env "$be")" "$GVIS" --directory "$DIR" \
              --iterations 1 --kernel after --prime --prime-wait flag --read "$read" 2>&1)
        # ROCm 7.15 logs "Barrier-AND", 7.1 logs "BarrierAND"
        seq=$(grep -oE '(Dispatch|Barrier-?AND|Barrier-?OR|Marker)[^,]*Header = 0x[0-9a-f]+' <<<"$out" |
              sed -E 's/ Header = /=/' | tr '\n' ' ')
        note "$be --read $read: $seq-> $(gvis_verdict "$out")"
    done
fi

# ---- hipbar: HIP-only repro of Failure 1 (CPU reads VRAM through the BAR during hipMemcpy) ---
if has_section hipbar; then
    section "hipbar: hip-bar-read-stale (no hipFile), stale iterations of 300 per size"
    gpu_busy_check hipbar
    if [ -x "$OUT/bin/hip-bar-read-stale" ] || run probe-build g++ -O2 -std=c++20 -D__HIP_PLATFORM_AMD__ -I"$ROCM/include" \
        "$HERE/src/hip-bar-read-stale.cpp" -o "$OUT/bin/hip-bar-read-stale" -L"$ROCM/lib" -lamdhip64 -pthread \
        -Wl,-rpath,"$ROCM/lib" >/dev/null; then
        for readers in 1 2 7 20; do
            for mem in coarse fine uncached; do
                for kick in 0 1; do
                    kick_env=()
                    [ "$kick" = 1 ] && kick_env=(KICK=1)
                    out=$(run hipbar env READERS=$readers MEM=$mem "${kick_env[@]}" "$OUT/bin/hip-bar-read-stale")
                    while read -r line; do
                        note "$(printf 'READERS=%-2s MEM=%-8s KICK=%s %s' "$readers" "$mem" "$kick" "$line")"
                    done < <(grep 'bytes:' <<<"$out")
                done
            done
        done
    else
        note "BUILD FAILED: hip-bar-read-stale"
    fi
fi

# ---- hdpprobe: write-only HDP flush between hipFileRead and the checking kernel (opt-in) -----
if has_section hdpprobe; then
    section "hdpprobe: --kernel after --prime with a write-only HDP flush after hipFileRead"
    if [ "$ALLOW_HDP" != 1 ]; then
        note "skipped: needs --allow-hdp-flush"
    else
        gpu_busy_check hdpprobe
        if [ ! -x "$OUT/bin/gvis-hdpflush" ]; then
            run probe-build "$HIPCXX" -std=c++20 -O1 -g $EXTRA_FLAGS "${HIPFILE_CFLAGS[@]}" -I"$SRC/test/amd_detail" \
                -c "$HERE/src/gpu-vis-hdpflush-probe.cpp" -o "$OUT/bin/gvis-hdpflush.o" >/dev/null &&
            run probe-build "$HIPCXX" -x hip --offload-arch="$ARCH" -std=c++20 -O2 $EXTRA_FLAGS -I"$SRC/test/amd_detail" \
                -c "$SRC/test/amd_detail/hipfileread-gpu-visibility-kernel.hip" -o "$OUT/bin/gvis-kernel.o" >/dev/null &&
            run probe-build "$HIPCXX" --hip-link --offload-arch="$ARCH" $EXTRA_FLAGS "$OUT/bin/gvis-hdpflush.o" \
                "$OUT/bin/gvis-kernel.o" -o "$OUT/bin/gvis-hdpflush" "${HIPFILE_LIBS[@]}" -L"$ROCM/lib" -lamdhip64 \
                -Wl,-rpath,"$ROCM/lib" >/dev/null || note "BUILD FAILED: gvis-hdpflush"
        fi
        if [ -x "$OUT/bin/gvis-hdpflush" ]; then
            be=${BACKENDS[0]}
            for spec in "sleep 0" "sleep 1" "flag 0" "flag 1" "sync 0"; do
                set -- $spec
                flush_env=()
                [ "$2" = 1 ] && flush_env=(PROBE_HDP_FLUSH=1)
                out=$(run hdpprobe env "$(backend_env "$be")" "${flush_env[@]}" "$OUT/bin/gvis-hdpflush" \
                      --directory "$DIR" --iterations "$GPU_ITERS" --kernel after --prime --prime-wait "$1")
                note "$(printf '%-8s --prime-wait %-5s HDP flush=%s: %s' "$be" "$1" "$2" "$(gvis_verdict "$out")")"
            done
        fi
    fi
fi

# ---- a1: barprobe configs, per-reader timing with background load, instrumented Kurt test ----
if has_section a1; then
    section "a1: barprobe (Failure 1 interventions), $REPS reps per config file"
    gpu_busy_check a1
    hip_build barprobe "$HERE/src/a1/barprobe.cpp" -DA1_WITH_HSA -I"$HERE/src/a1" -lhsa-runtime64 -pthread &&
    hip_build bgload "$HERE/src/a1/bgload.cpp" -DA1_WITH_HSA -I"$HERE/src/a1" -lhsa-runtime64 -pthread &&
    for cfg in "$HERE"/src/a1/cfg*.txt; do
        note "-- $(basename "$cfg")"
        for ((rep = 1; rep <= REPS; rep++)); do
            while IFS= read -r line; do
                [[ -z $line || $line == \#* ]] && continue
                if hdp_line "$line" && [ "$ALLOW_HDP" != 1 ]; then note "r$rep SKIPPED (HDP): $line"; continue; fi
                out=$(eval "run a1 env $line $QUICK_ITERS \"$OUT/bin/barprobe\"" | grep -v '^#' | tail -2)
                note "r$rep $(tr '\n' ' ' <<<"$out")"
            done <"$cfg"
        done
    done
    if [ -x "$OUT/bin/barprobe" ] && [ -x "$OUT/bin/bgload" ]; then
        note "-- per-reader timing, idle vs background GPU load (dumps in $OUT/a1-dumps)"
        mkdir -p "$OUT/a1-dumps"
        for readers in 4 7 20; do
            out=$(run a1 env LABEL=idle-R$readers READERS=$readers WAITALL=1 ITERS=300 DUMP="$OUT/a1-dumps/idle-R$readers.csv" "$OUT/bin/barprobe" | grep -v '^#' | tail -1)
            note "$out"
        done
        for mode in sdma longkernel kernels; do
            [ "$DRY" = 1 ] || { MODE=$mode SECS=6 "$OUT/bin/bgload" >>"$OUT/logs/a1.log" 2>&1 & }
            bg=$!
            sleep 1.5
            out=$(run a1 env LABEL=bg-$mode-R7 READERS=7 WAITALL=1 ITERS=300 DUMP="$OUT/a1-dumps/bg-$mode-R7.csv" "$OUT/bin/barprobe" | grep -v '^#' | tail -1)
            note "$out"
            [ "$DRY" = 1 ] || wait "$bg"
        done
    fi
    note "-- Kurt's test instrumented: PROBE_COUNT=1 counts stale watcher events; PROBE_PRE/POST interventions"
    if [ ! -x "$OUT/bin/hfcv-instr" ]; then
        run probe-build python3 "$HERE/src/a1/make-instrumented.py" "$SRC/test/amd_detail/hipfileread-cpu-visibility.cpp" \
            "$OUT/bin/hfcv-instr.cpp" >/dev/null &&
        run probe-build "$HIPCXX" -x hip --offload-arch="$ARCH" -O1 -g -std=c++20 $EXTRA_FLAGS -DA1_WITH_HSA "${HIPFILE_CFLAGS[@]}" \
            -I"$HERE/src/a1" "$OUT/bin/hfcv-instr.cpp" -o "$OUT/bin/hfcv-instr" "${HIPFILE_LIBS[@]}" -L"$ROCM/lib" \
            -lhsa-runtime64 -pthread -Wl,-rpath,"$ROCM/lib" >/dev/null || note "BUILD FAILED: hfcv-instr"
    fi
    if [ -x "$OUT/bin/hfcv-instr" ]; then
        for ((rep = 1; rep <= REPS; rep++)); do
            for post in none devsync devsync,memset,devsync hdpmemnoread evt sync0 aqlacqsys aqlrelsys; do
                if hdp_line "$post" && [ "$ALLOW_HDP" != 1 ]; then continue; fi
                out=$(run a1 env A1_AQL=1 PROBE_COUNT=1 PROBE_POST=$post HIPFILE_FORCE_COMPAT_MODE=true "$OUT/bin/hfcv-instr" \
                      --directory "$DIR" --iterations 300 --cpus 8 --buffer-size 64K | grep PROBE)
                note "r$rep fallback 64K cpus8 post=$post: ${out#PROBE }"
            done
            if [ "$ALLOW_HDP" = 1 ]; then
                out=$(run a1 env PROBE_COUNT=1 PROBE_PRE=hdpmemnoread HIPFILE_FORCE_COMPAT_MODE=true "$OUT/bin/hfcv-instr" \
                      --directory "$DIR" --iterations 300 --cpus 8 --buffer-size 64K | grep PROBE)
                note "r$rep fallback 64K cpus8 pre=hdpmemnoread: ${out#PROBE }"
            fi
            if [ "$FASTPATH" = 1 ]; then
                for post in none hdpmemnoread devsync,memset,devsync; do
                    if hdp_line "$post" && [ "$ALLOW_HDP" != 1 ]; then continue; fi
                    for cpus in 8 "$(nproc)"; do
                        out=$(run a1 env PROBE_COUNT=1 PROBE_POST=$post HIPFILE_ALLOW_COMPAT_MODE=false "$OUT/bin/hfcv-instr" \
                              --directory "$DIR" --iterations 1000 --cpus "$cpus" --buffer-size 64K | grep -E 'PROBE|rror' | head -1)
                        note "r$rep fastpath 64K cpus$cpus post=$post: ${out#PROBE }"
                    done
                done
            fi
        done
    fi
fi

# ---- a2: stale-probe batches (Failure 2: which GPU cache holds stale data, which packets fix it) -
if has_section a2; then
    section "a2: stale-probe batches b01-b11"
    case $ARCH in
        gfx10* | gfx11*)
            gpu_busy_check a2
            hip_build stale-probe "$HERE/src/a2/stale-probe.hip" -lhsa-runtime64
            hip_build stale-probe-hf "$HERE/src/a2/stale-probe.hip" -DWITH_HIPFILE -isystem "$SRC/include" \
                "${HIPFILE_LIBS[@]}" -lhsa-runtime64
            for batch in "$HERE"/src/a2/b*.txt; do
                note "-- $(basename "$batch")"
                while IFS= read -r line; do
                    [ -z "$line" ] && continue
                    if [[ $line == \#* ]]; then note "$line"; continue; fi
                    if hdp_line "$line" && [ "$ALLOW_HDP" != 1 ]; then note "SKIPPED (HDP): $line"; continue; fi
                    if [[ $line == *'--writer hipfile'* && $line == *HIPFILE_ALLOW_COMPAT_MODE=false* && $FASTPATH != 1 ]]; then
                        note "SKIPPED (no fastpath): $line"; continue
                    fi
                    envs=() args=() bin=$OUT/bin/stale-probe
                    for w in $line; do
                        if [[ ${#args[@]} -eq 0 && $w == PROBE=* ]]; then bin=$OUT/bin/$(basename "${w#PROBE=}")
                        elif [[ ${#args[@]} -eq 0 && $w == HF_PATH=* ]]; then envs+=("HF_PATH=$DIR/a2-probe.bin")
                        elif [[ ${#args[@]} -eq 0 && $w == *=* ]]; then envs+=("$w")
                        else args+=("$w"); fi
                    done
                    out=$(run a2 env "${envs[@]}" "$bin" "${args[@]}" | grep -E '^RESULT|^POINTER|rror|failed')
                    note "${envs[*]} $(basename "$bin") ${args[*]}"
                    [ -n "$out" ] && note "    $(tr '\n' ' ' <<<"$out" | cut -c1-400)"
                done <"$batch"
            done
            rm -f "$DIR/a2-probe.bin"
            ;;
        *) note "skipped: stale-probe uses gfx10/gfx11 inline assembly (glc dlc loads, buffer_gl0_inv); this GPU is $ARCH" ;;
    esac
fi

# ---- c: f1probe (Failure 1 discriminators) and f2probe (Failure 2 discriminators) ------------
if has_section c; then
    section "c: f1probe and f2probe configs"
    gpu_busy_check c
    hip_build f1probe "$HERE/src/c/f1probe.hip" -msse4.1 -lamdhip64
    hip_build f2probe "$HERE/src/c/f2probe.hip" -lamdhip64
    for probe in f1 f2; do
        [ -x "$OUT/bin/${probe}probe" ] || continue
        note "-- ${probe}probe"
        while IFS= read -r line; do
            [[ -z $line || $line == \#* ]] && continue
            if hdp_line "$line" && [ "$ALLOW_HDP" != 1 ]; then note "SKIPPED (HDP): $line"; continue; fi
            out=$(eval "run c env $line \"$OUT/bin/${probe}probe\"" | grep -vE '^\s*$' | tail -3)
            note "$line => $(tr '\n' ' ' <<<"$out" | cut -c1-300)"
        done <"$HERE/src/c/$probe-configs.txt"
    done
fi

health after
section "done"
note "summary: $SUMMARY"
note "full logs: $OUT/logs/ (per section), $OUT/run.log (everything)"
