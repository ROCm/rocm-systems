#!/bin/bash
set -euo pipefail

RCCL_LIB_DIR="/work/lmeadows/split-compile/projects/rccl/build"
ROCPROFV3="/opt/rocm/bin/rocprofv3"
MPIRUN="/work/lmeadows/openmpi/bin/mpirun"
TEST_DIR="/work/lmeadows/split-compile/projects/rccl-tests/build"
DATA_DIR="/work/lmeadows/data"
NP=8
STEP_FACTOR=2

BENCH_BEGIN_SIZE="${BENCH_BEGIN_SIZE:-}"
BENCH_END_SIZE="${BENCH_END_SIZE:-1G}"
BENCH_DATATYPE="${BENCH_DATATYPE:-}"
BENCH_TESTS="${BENCH_TESTS:-}"

# Minimum -b value per test (used when BENCH_BEGIN_SIZE is not set).
# Operations that compute (count/nranks) & -(16/eltSize) yield zero when
# count < nranks*16.  With 8 ranks that threshold is 128 bytes; use 256
# for headroom.  alltoallv additionally requires count >= nranks^2/2 = 32,
# which 256 already covers.
declare -A MIN_BYTES=(
    [all_reduce_perf]=8
    [all_reduce_bias_perf]=8
    [broadcast_perf]=8
    [reduce_perf]=8
    [sendrecv_perf]=8
    [all_gather_perf]=256
    [reduce_scatter_perf]=256
    [scatter_perf]=256
    [gather_perf]=256
    [alltoall_perf]=256
    [alltoallv_perf]=256
    [hypercube_perf]=256
)

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_DIR="${DATA_DIR}/${TIMESTAMP}"
mkdir -p "$RUN_DIR"

# Per-rank wrapper: each MPI process invokes rocprofv3 with a
# rank-specific output directory derived from OMPI_COMM_WORLD_RANK.
WRAPPER="$RUN_DIR/.rocprof_wrapper.sh"
cat > "$WRAPPER" << 'WRAPPER_EOF'
#!/bin/bash
RANK=${OMPI_COMM_WORLD_RANK:-unknown}
exec /opt/rocm/bin/rocprofv3 \
    --kernel-trace --marker-trace \
    -d "${ROCP_OUT_DIR}/rank_${RANK}" \
    -- "$@"
WRAPPER_EOF
chmod +x "$WRAPPER"

run_bench() {
    local name=$1
    local bench="${TEST_DIR}/${name}"
    local begin_b="${BENCH_BEGIN_SIZE:-${MIN_BYTES[$name]}}"
    local end_b="${BENCH_END_SIZE}"
    local out_dir="${RUN_DIR}/${name%_perf}"

    local extra_args=()
    if [[ -n "$BENCH_DATATYPE" ]]; then
        extra_args+=(-d "$BENCH_DATATYPE")
    fi

    echo "========================================"
    echo "  Running: ${name}  (-b ${begin_b} -e ${end_b} -f ${STEP_FACTOR}${BENCH_DATATYPE:+ -d ${BENCH_DATATYPE}})"
    echo "  Output:  ${out_dir}/"
    echo "========================================"

    mkdir -p "$out_dir"

    local rc=0
    env \
        LD_LIBRARY_PATH="${RCCL_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ROCP_OUT_DIR="$out_dir" \
        "$MPIRUN" -np "$NP" "$WRAPPER" \
            "$bench" -b "$begin_b" -e "$end_b" -f "$STEP_FACTOR" \
            "${extra_args[@]}" \
        2>&1 | tee "${out_dir}/stdout.log" || rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "WARNING: ${name} exited with code ${rc}"
    fi
    echo ""
}

ALL_BENCHMARKS=(
    all_reduce_perf
    all_reduce_bias_perf
    all_gather_perf
    alltoall_perf
    alltoallv_perf
    broadcast_perf
    gather_perf
    hypercube_perf
    reduce_perf
    reduce_scatter_perf
    scatter_perf
    sendrecv_perf
)

if [[ -n "$BENCH_TESTS" ]]; then
    IFS=',' read -ra BENCHMARKS <<< "$BENCH_TESTS"
else
    BENCHMARKS=("${ALL_BENCHMARKS[@]}")
fi

echo "rccl-tests benchmark suite"
echo "Date:     $(date --iso-8601=seconds)"
echo "Host:     $(hostname)"
echo "NP:       ${NP}"
echo "RCCL lib: ${RCCL_LIB_DIR}/librccl.so"
echo "Run dir:  ${RUN_DIR}"
echo "Begin:    ${BENCH_BEGIN_SIZE:-<per-test default>}"
echo "End:      ${BENCH_END_SIZE}"
echo "Datatype: ${BENCH_DATATYPE:-<default>}"
echo "Tests:    ${BENCHMARKS[*]}"
echo ""

for bench in "${BENCHMARKS[@]}"; do
    if [[ ! -x "${TEST_DIR}/${bench}" ]]; then
        echo "SKIP: ${bench} (not found or not executable)"
        continue
    fi
    run_bench "$bench"
done

echo "All benchmarks complete.  Data collected in ${RUN_DIR}/"
