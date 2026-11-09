#!/usr/bin/env bash
# Parity step 1: run an app under rocprofv3 --hip-trace and an LTTng session recording the
# rocm_hip probes, in the same process.
#
# Environment: clr-trace-build:7.14 image; TheRock nightly 10.3 runtime (ROCr, comgr 3.5,
# rocprofiler-sdk, rocprofv3) mounted at /opt/nr; only libamdhip64 replaced by the CLR build
# with HIP_ENABLE_ROCM_TRACE=ON (#15122).
#
# Usage: run_both.sh <out-dir> <app> [args...]   (paths under the build dir)
set -euo pipefail

B=/home/gobhardw/workarea/clr-trace-build
N=/tmp/nightly-10.3/root/opt/rocm/core-10.3
J=/home/gobhardw/workarea/oop-poc/experimental/oop-tracing-poc/correlator
S=/home/gobhardw/workarea/clr-trace/projects/clr/hipamd/src/trace
OUT=${1:?out dir}
shift

rm -rf "${OUT}"
mkdir -p "${OUT}"

docker run --rm --device /dev/kfd --device /dev/dri --group-add video \
    --security-opt seccomp=unconfined -v "${B}:${B}" -v "${N}:/opt/nr:ro" \
    -v "${J}:${J}:ro" -v "${S}:${S}:ro" \
    clr-trace-build:7.14-parity bash -c "
set -e
export LD_LIBRARY_PATH=${B}/hipamd/lib:/opt/nr/lib:/opt/nr/lib/rocm_sysdeps/lib
lttng-sessiond --no-kernel --daemonize >/dev/null 2>&1
sleep 1
lttng create s --output=${OUT}/lttng >/dev/null
lttng enable-event -u 'rocm_hip:*' >/dev/null
lttng add-context -u -t vpid -t vtid >/dev/null
lttng start >/dev/null
/opt/nr/bin/rocprofv3 --hip-trace --output-format csv -d ${OUT}/rocprofv3 -o run -- $* \
    > ${OUT}/rocprofv3.log 2>&1
lttng stop >/dev/null
lttng destroy >/dev/null
babeltrace2 ${OUT}/lttng > ${OUT}/lttng.txt 2> ${OUT}/lttng.err || true
/usr/bin/python3.10 ${J}/join.py ${OUT} --rocprofv3-csv=${OUT}/probes_hip_api_trace.csv \
    --probes-src=${S}/hip_trace_probes.cpp > ${OUT}/join.log
chmod -R a+rwX ${OUT}
"
echo "rocprofv3 rows: $(($(wc -l < "${OUT}/rocprofv3/run_hip_api_trace.csv") - 1))"
echo "probe events:   $(wc -l < "${OUT}/lttng.txt")"
echo "probe calls:    $(($(wc -l < "${OUT}/probes_hip_api_trace.csv") - 1))"
cat "${OUT}/join.log"
echo "LTTng discarded/lost: $(grep -ci "discard\|lost" "${OUT}/lttng.err" || true)"
# Step 6: compare. __hipUnregisterFatBinary runs at exit, after rocprofv3 has finalized.
python3 "${J}/compare_rocprofv3.py" "${OUT}/rocprofv3/run_hip_api_trace.csv" \
    "${OUT}/probes_hip_api_trace.csv" --allow-extra=__hipUnregisterFatBinary \
    | tee "${OUT}/compare.txt"
