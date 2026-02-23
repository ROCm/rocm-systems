#!/bin/bash
set -euo pipefail

RCCL_LIB_DIR="/work/lmeadows/split-compile/projects/rccl/build"
ROCPROFV3="/opt/rocm/bin/rocprofv3"
MPIRUN="/work/lmeadows/openmpi/bin/mpirun"
TEST_DIR="/work/lmeadows/split-compile/projects/rccl-tests/build"
DATA_DIR="/work/lmeadows/data"
NP=8
MAX_BYTES="1G"
STEP_FACTOR=2

# Minimum -b value per test.
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

mkdir -p "$DATA_DIR"

# Per-rank wrapper: each MPI process invokes rocprofv3 with a
# rank-specific output directory derived from OMPI_COMM_WORLD_RANK.
WRAPPER="$DATA_DIR/.rocprof_wrapper.sh"
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
    local min_b=${MIN_BYTES[$name]}
    local out_dir="${DATA_DIR}/${name%_perf}"

    echo "========================================"
    echo "  Running: ${name}  (-b ${min_b} -e ${MAX_BYTES} -f ${STEP_FACTOR})"
    echo "  Output:  ${out_dir}/"
    echo "========================================"

    rm -rf "$out_dir"
    mkdir -p "$out_dir"

    local rc=0
    env \
        LD_LIBRARY_PATH="${RCCL_LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
        ROCP_OUT_DIR="$out_dir" \
        "$MPIRUN" -np "$NP" "$WRAPPER" \
            "$bench" -b "$min_b" -e "$MAX_BYTES" -f "$STEP_FACTOR" \
        2>&1 | tee "${out_dir}/stdout.log" || rc=$?

    if [[ $rc -ne 0 ]]; then
        echo "WARNING: ${name} exited with code ${rc}"
    fi
    echo ""
}

BENCHMARKS=(
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

echo "rccl-tests benchmark suite"
echo "Date:     $(date --iso-8601=seconds)"
echo "Host:     $(hostname)"
echo "NP:       ${NP}"
echo "RCCL lib: ${RCCL_LIB_DIR}/librccl.so"
echo "Data dir: ${DATA_DIR}"
echo ""

for bench in "${BENCHMARKS[@]}"; do
    if [[ ! -x "${TEST_DIR}/${bench}" ]]; then
        echo "SKIP: ${bench} (not found or not executable)"
        continue
    fi
    run_bench "$bench"
done

echo "All benchmarks complete.  Data collected in ${DATA_DIR}/"
