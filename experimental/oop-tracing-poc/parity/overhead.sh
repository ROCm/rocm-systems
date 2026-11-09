#!/usr/bin/env bash
# Parity step 11: run an application separately in each mode, nothing else in the process.
#
#   overhead.sh <out-dir> <reps> <app> [args...]
#
# Modes (wall time per run, median of <reps>):
#   baseline     nightly libamdhip64 (no probes), no tracing
#   probes-idle  probe libamdhip64, no LTTng session          (cost of the probes when off)
#   probes-rec   probe libamdhip64, LTTng session recording   (cost of recording)
#   rocprofv3    nightly libamdhip64, rocprofv3 --hip-trace   (today's tool)
# The last run of probes-rec and rocprofv3 keeps its trace, converted to hip_api_trace.csv.
set -euo pipefail

B=/home/gobhardw/workarea/clr-trace-build
N=/tmp/nightly-10.3/root/opt/rocm/core-10.3
J=/home/gobhardw/workarea/oop-poc/experimental/oop-tracing-poc/correlator
S=/home/gobhardw/workarea/clr-trace/projects/clr/hipamd/src/trace
HIPLIB=/home/gobhardw/workarea/clr-trace-build2/hipamd/lib/libamdhip64.so.7.18.0-0000000
HIP_REAL=$(basename "$(readlink -f "${N}/lib/libamdhip64.so.7")")
OUT=${1:?out dir}
REPS=${2:?reps}
shift 2
APP="$*"

docker run --rm -v "$(dirname "${OUT}"):/o" clr-trace-build:7.14-parity rm -rf "/o/$(basename "${OUT}")"
mkdir -p "${OUT}"

run_mode() {
    local mode=$1
    local mount=()
    if [[ ${mode} == probes-* ]]; then
        mount=(-v "${HIPLIB}:/opt/nr/lib/${HIP_REAL}:ro")
    fi
    docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
        --security-opt seccomp=unconfined -v "${B}:${B}" -v "${N}:/opt/nr:ro" \
        -v "${J}:${J}:ro" -v "${S}:${S}:ro" "${mount[@]}" \
        clr-trace-build:7.14-parity bash -c "
export LD_LIBRARY_PATH=/opt/nr/lib:/opt/nr/lib/rocm_sysdeps/lib
D=${OUT}/${mode}
mkdir -p \${D}
if [ ${mode} = probes-rec ]; then
    lttng-sessiond --no-kernel --daemonize >/dev/null 2>&1
    sleep 1
fi
for i in \$(seq ${REPS}); do
    if [ ${mode} = probes-rec ]; then
        rm -rf \${D}/lttng
        lttng create s --output=\${D}/lttng >/dev/null
        lttng enable-event -u 'rocm_hip:*' >/dev/null
        lttng add-context -u -t vpid -t vtid >/dev/null
        lttng start >/dev/null
    fi
    t0=\$(date +%s%N)
    if [ ${mode} = rocprofv3 ]; then
        rm -rf \${D}/rocprofv3
        /opt/nr/bin/rocprofv3 --hip-trace --output-format csv -d \${D}/rocprofv3 -o run -- ${APP} \
            > \${D}/app.log 2>&1 || echo app exit status \$? >> \${D}/app_status.txt
    else
        ${APP} > \${D}/app.log 2>&1 || echo app exit status \$? >> \${D}/app_status.txt
    fi
    t1=\$(date +%s%N)
    echo \$(( (t1 - t0) / 1000 )) >> \${D}/times_us.txt
    if [ ${mode} = probes-rec ]; then
        lttng stop >/dev/null
        lttng destroy >/dev/null
    fi
done
if [ ${mode} = probes-rec ]; then
    babeltrace2 \${D}/lttng > /dev/null 2> \${D}/lttng.err || true
    /usr/bin/python3.10 ${J}/join.py \${D} --rocprofv3-csv=\${D}/probes_hip_api_trace.csv \
        --probes-src=${S}/hip_trace_probes.cpp > \${D}/join.log
fi
chmod -R a+rwX ${OUT}
"
}

for mode in baseline probes-idle probes-rec rocprofv3; do
    run_mode "${mode}"
done

python3 - "${OUT}" <<'EOF'
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
med = {}
for mode in ["baseline", "probes-idle", "probes-rec", "rocprofv3"]:
    ts = [int(x) for x in (out / mode / "times_us.txt").read_text().split()]
    med[mode] = statistics.median(ts)
    status = out / mode / "app_status.txt"
    failed = f"   FAILED RUNS: {status.read_text().count('exit')}" if status.exists() else ""
    print(f"{mode:12s} median {med[mode] / 1e3:10.1f} ms  (runs: "
          f"{', '.join(f'{t / 1e3:.1f}' for t in ts)}){failed}")
for mode in ["probes-idle", "probes-rec", "rocprofv3"]:
    print(f"{mode:12s} vs baseline: {100 * (med[mode] / med['baseline'] - 1):+6.1f} %")
EOF
