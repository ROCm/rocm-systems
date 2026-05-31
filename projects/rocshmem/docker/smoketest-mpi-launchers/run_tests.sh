#!/bin/bash
# rocSHMEM MPI portability CI test runner.
#
# Each launcher section (ompi, mpich, slurm) contains:
#   - One driver.sh validation case (tests launcher identification)
#   - Direct invocation cases covering all relevant build × uuid × backend combos
#
# Expected outcome table (C=CORRECT/expected pass, X=expected fail, SKIP=not run):
#
#  build      | backend | uuid | launcher        | expected | Comment
#  -----------+---------+------+-----------------+----------+-------------------------------
#  -- mpiexec/prte --
#  ompi-auto  | ipc     |  0   | mpiexec/prte    |    C     |
#  ompi-auto  | ipc     |  1   | mpiexec/prte    |    C     |
#  ompi-on    | ipc     |  0   | mpiexec/prte    |    C     |
#  ompi-on    | ipc     |  1   | mpiexec/prte    |    C     |
#  ompi-auto  | ro      |  0   | mpiexec/prte    |    C     |
#  ompi-auto  | ro      |  1   | mpiexec/prte    |    C     |
#  ompi-on    | ro      |  0   | mpiexec/prte    |    C     |
#  ompi-on    | ro      |  1   | mpiexec/prte    |    C     |
#  -- mpiexec/hydra (uuid=1 omitted — Hydra has no PMIx server) --
#  mpich-auto | ipc     |  0   | mpiexec/hydra   |    C     |
#  mpich-off  | ipc     |  0   | mpiexec/hydra   |    X     | dlopen→MPICH→MPILIB_INCOMPATIBLE
#  mpich-auto | ro      |  0   | mpiexec/hydra   |    C     | HAVE_EXTERNAL_MPI=ON, MPICH MPI used
#  mpich-off  | ro      |  0   | mpiexec/hydra   |    X     | dlopen→MPICH→MPILIB_INCOMPATIBLE
#  -- srun --
#  ompi-auto  | ipc     |  0   | srun --mpi=pmix |    C     |
#  ompi-auto  | ipc     |  1   | srun --mpi=pmix |    C     |
#  ompi-on    | ipc     |  0   | srun --mpi=pmix |    C     |
#  ompi-on    | ipc     |  1   | srun --mpi=pmix |    C     |
#  mpich-auto | ipc     |  0   | srun --mpi=pmi2 |    C     |
#  mpich-auto | ipc     |  1   | srun --mpi=pmix |    C     | srun provides PMIx server
#  mpich-off  | ipc     |  0   | srun --mpi=pmi2 |    X     | dlopen→MPICH→MPILIB_INCOMPATIBLE
#  mpich-off  | ipc     |  1   | srun --mpi=pmix |    C     | TcpBootstrap+IPC, no MPI needed
#  ompi-auto  | ro      |  0   | srun --mpi=pmix |    C     |
#  ompi-auto  | ro      |  1   | srun --mpi=pmix |    C     |
#  ompi-on    | ro      |  0   | srun --mpi=pmix |    C     |
#  ompi-on    | ro      |  1   | srun --mpi=pmix |    C     |
#  mpich-auto | ro      |  0   | srun --mpi=pmi2 |    C     | HAVE_EXTERNAL_MPI=ON, MPICH MPI used
#  mpich-auto | ro      |  1   | —               |  SKIP    | RO needs --mpi=pmi2, uuid needs --mpi=pmix; mutually exclusive
#  mpich-off  | ro      |  0   | srun --mpi=pmi2 |    X     | dlopen→MPICH→MPILIB_INCOMPATIBLE
#  mpich-off  | ro      |  1   | —               |  SKIP    | RO needs --mpi=pmi2, uuid needs --mpi=pmix; mutually exclusive
#
# Usage (inside the container, GPU devices must be accessible):
#   run_tests.sh
#
# Required docker run flags:
#   --network host --ipc host --shm-size 64G
#   --device /dev/kfd --device /dev/dri --group-add video
#   --privileged   (needed for the Slurm section)
#
# Optional env vars:
#   VARIANTS   space-separated subset of sections to run:
#              ompi  mpich  slurm   (default: all three)
#   VERBOSE    set to 1 to pipe individual test output to stdout (default: 0)
#   ROCSHMEM_DEBUG_LEVEL  (default: warn)

set -euo pipefail

ROCSHMEM_SRC="${ROCSHMEM_SRC:-/workspace/src/projects/rocshmem}"
DRIVER="${ROCSHMEM_SRC}/scripts/functional_tests/driver.sh"
LOG_BASE="/tmp/rocshmem-ci-logs"
NRANKS=2
VERBOSE=${VERBOSE:-0}

OMPI_BIN=/opt/ompi/bin
MPICH_BIN=/opt/mpich/bin
OMPI_LIB=/opt/ompi/lib
MPICH_LIB=/opt/mpich/lib
UCX_LIB=/opt/ucx/lib
SYSTEM_LIB=/usr/lib/x86_64-linux-gnu

# Test binary args for direct invocation
TESTER_ARGS="-a 26 -w 1 -z 256"

mkdir -p "${LOG_BASE}"

# --------------------------------------------------------------------------
# Result tracking
# --------------------------------------------------------------------------
declare -a RESULTS=()   # entries: "CORRECT|INCORRECT|label"
OVERALL_RC=0

record() {
    local status="$1"   # CORRECT or INCORRECT
    local label="$2"
    RESULTS+=("${status}|${label}")
    if [[ "$status" = "INCORRECT" ]]; then OVERALL_RC=1; fi
}

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
run_driver() {
    local label="$1"; local app="$2"; local log_dir="${LOG_BASE}/${label}"
    mkdir -p "$log_dir"
    echo "  > ${label} (driver.sh)"
    set +e
    if [[ "$VERBOSE" = "1" ]]; then
        bash "${DRIVER}" "$app" pingpong "$log_dir" 2>&1 | tee "${log_dir}/out.log"; local rc=${PIPESTATUS[0]}
    else
        bash "${DRIVER}" "$app" pingpong "$log_dir" >"${log_dir}/out.log" 2>&1; local rc=$?
    fi
    set -e
    if [[ $rc -eq 0 ]]; then
        echo "    CORRECT: ${label}"
        record CORRECT "${label}"
    else
        echo "    INCORRECT: ${label} — exit ${rc}. Log: ${log_dir}/out.log"
        record INCORRECT "${label}"
    fi
}

# run_direct LABEL APP LAUNCHER_CMD... -- ENV_VAR=val... -- EXPECTED (pass|fail)
run_direct() {
    local label="$1"; shift
    local app="$1"; shift
    local -a launcher_cmd=()
    while [[ "$1" != "--" ]]; do launcher_cmd+=("$1"); shift; done; shift
    local -a env_pairs=()
    while [[ "$1" != "--" ]]; do env_pairs+=("$1"); shift; done; shift
    local expected="$1"   # "pass" or "fail"

    local log="${LOG_BASE}/${label}.log"
    echo "  > ${label} (expect: ${expected^^})"

    local -a env_cmd=(env
        ROCSHMEM_DEBUG_LEVEL="${ROCSHMEM_DEBUG_LEVEL:-warn}"
        OMPI_ALLOW_RUN_AS_ROOT=1
        OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)
    for kv in "${env_pairs[@]}"; do env_cmd+=("$kv"); done

    [[ "$VERBOSE" = "1" ]] && echo "  CMD: ${env_cmd[*]} ${launcher_cmd[*]} $app ${TESTER_ARGS}"

    set +e
    if [[ "$VERBOSE" = "1" ]]; then
        "${env_cmd[@]}" "${launcher_cmd[@]}" "$app" ${TESTER_ARGS} 2>&1 | tee "$log"
    else
        "${env_cmd[@]}" "${launcher_cmd[@]}" "$app" ${TESTER_ARGS} >"$log" 2>&1
    fi
    local rc=${PIPESTATUS[0]}
    set -e

    if [[ "$expected" = "pass" ]]; then
        if [[ $rc -eq 0 ]]; then
            echo "    CORRECT: ${label}"
            record CORRECT "${label}"
        else
            echo "    INCORRECT: ${label} — expected exit 0, got ${rc}. Log: ${log}"
            tail -5 "$log" | sed 's/^/      /'
            record INCORRECT "${label}"
        fi
    else
        local log_content; log_content=$(cat "$log" 2>/dev/null || true)
        if [[ $rc -ne 0 ]] && echo "$log_content" | grep -qE \
            "not Open MPI|MPILIB_INCOMPATIBLE|RO backend cannot run|requires exactly two processes|PMIx_Init failed"; then
            echo "    CORRECT: ${label} — correctly rejected"
            record CORRECT "${label}"
        elif [[ $rc -ne 0 ]]; then
            echo "    INCORRECT: ${label} — non-zero exit (${rc}) but expected error string not found. Log: ${log}"
            tail -5 "$log" | sed 's/^/      /'
            record INCORRECT "${label}"
        else
            echo "    INCORRECT: ${label} — expected non-zero exit but got 0"
            record INCORRECT "${label}"
        fi
    fi
}

# --------------------------------------------------------------------------
# VARIANTS filter
# --------------------------------------------------------------------------
RUN_OMPI=1; RUN_MPICH=1; RUN_SLURM=1
if [[ -n "${VARIANTS:-}" ]]; then
    RUN_OMPI=0; RUN_MPICH=0; RUN_SLURM=0
    for v in $VARIANTS; do
        case "$v" in
            ompi)  RUN_OMPI=1  ;;
            mpich) RUN_MPICH=1 ;;
            slurm) RUN_SLURM=1 ;;
            *) echo "Unknown variant '$v'; valid: ompi mpich slurm" ;;
        esac
    done
fi

# --------------------------------------------------------------------------
# Slurm startup (idempotent)
# --------------------------------------------------------------------------
start_slurm_once() {
    if ! sinfo &>/dev/null 2>&1; then
        echo "Starting single-node Slurm cluster..."
        bash /usr/local/bin/start_slurm.sh
    else
        echo "Slurm already running."
    fi
}

# ==========================================================================
# Open MPI — mpiexec/prte
# ==========================================================================
if [[ $RUN_OMPI -eq 1 ]]; then
    echo ""
    echo "========================================================================"
    echo " Section 1: Open MPI mpiexec/prte — ompi-auto and ompi-on"
    echo "========================================================================"

    export LD_LIBRARY_PATH="${OMPI_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH}"
    MPIRUN=("${OMPI_BIN}/mpiexec" -n ${NRANKS} -mca pml ucx -mca osc ucx --map-by numa)

    echo ""
    echo "  --- Section 1.1: driver.sh — mpiexec/prte launcher identification ---"
    APP="${ROCSHMEM_SRC}/build-ompi-auto/tests/functional_tests/rocshmem_functional_tests"
    if [[ -x "$APP" ]]; then
        export LAUNCHER=mpiexec ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc
        export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        export PATH="${OMPI_BIN}:${PATH}"
        run_driver "driver-ompi-mpiexec" "$APP"
        unset LAUNCHER ROCSHMEM_TEST_UUID ROCSHMEM_BACKEND OMPI_ALLOW_RUN_AS_ROOT OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
        export PATH="${PATH#${OMPI_BIN}:}"
    else
        echo "  SKIP: ompi-auto binary not found"
    fi

    echo ""
    echo "  --- Section 1.2: direct — mpiexec/prte, ipc, uuid=0 ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "${build}-ipc-uuid0" "$APP" "${MPIRUN[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc -- pass
    done

    echo ""
    echo "  --- Section 1.3: direct — mpiexec/prte, ipc, uuid=1 ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "${build}-ipc-uuid1" "$APP" "${MPIRUN[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ipc -- pass
    done

    echo ""
    echo "  --- Section 1.4: direct — mpiexec/prte, ro, uuid=0 ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "${build}-ro-uuid0"  "$APP" "${MPIRUN[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  -- pass
    done

    echo ""
    echo "  --- Section 1.5: direct — mpiexec/prte, ro, uuid=1 ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "${build}-ro-uuid1"  "$APP" "${MPIRUN[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ro  -- pass
    done

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${OMPI_LIB}:${UCX_LIB}:}"
fi

# ==========================================================================
# MPICH — mpiexec/hydra
# ==========================================================================
if [[ $RUN_MPICH -eq 1 ]]; then
    echo ""
    echo "========================================================================"
    echo " Section 2: MPICH mpiexec/hydra — mpich-auto and mpich-off"
    echo "========================================================================"

    export LD_LIBRARY_PATH="${MPICH_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH}"
    MPIEXEC=("${MPICH_BIN}/mpiexec" -n ${NRANKS})

    echo ""
    echo "  --- Section 2.1: driver.sh — mpiexec/hydra launcher identification ---"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    if [[ -x "$APP" ]]; then
        export LAUNCHER=mpiexec ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc
        export PATH="${MPICH_BIN}:${PATH}"
        run_driver "driver-mpich-mpiexec" "$APP"
        unset LAUNCHER ROCSHMEM_TEST_UUID ROCSHMEM_BACKEND
        export PATH="${PATH#${MPICH_BIN}:}"
    else
        echo "  SKIP: mpich-auto binary not found"
    fi

    echo ""
    echo "  --- Section 2.2: direct — mpiexec/hydra, ipc, uuid=0 ---"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "mpich-auto-ipc-uuid0" "$APP" "${MPIEXEC[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc -- pass || echo "  SKIP: mpich-auto binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-off/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "mpich-off-ipc-uuid0"  "$APP" "${MPIEXEC[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc -- fail || echo "  SKIP: mpich-off binary not found"

    echo ""
    echo "  --- Section 2.3: direct — mpiexec/hydra, ro, uuid=0 ---"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "mpich-auto-ro-uuid0"  "$APP" "${MPIEXEC[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  -- pass || echo "  SKIP: mpich-auto binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-off/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "mpich-off-ro-uuid0"   "$APP" "${MPIEXEC[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  -- fail || echo "  SKIP: mpich-off binary not found"
    # uuid=1 omitted for mpiexec: Hydra has no PMIx server

    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${MPICH_LIB}:${UCX_LIB}:}"
fi

# ==========================================================================
# Slurm — srun
# ==========================================================================
if [[ $RUN_SLURM -eq 1 ]]; then
    echo ""
    echo "========================================================================"
    echo " Section 3: Slurm srun — all builds"
    echo "========================================================================"

    # SYSTEM_LIB first so srun's mpi/pmix_v5 plugin finds the system libpmix
    export LD_LIBRARY_PATH="${SYSTEM_LIB}:${OMPI_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH}"
    start_slurm_once

    SRUN_PMIX=(srun -n ${NRANKS} --mpi=pmix)   # PMIx server — uuid=1 and ompi builds
    SRUN_PMI2=(srun -n ${NRANKS} --mpi=pmi2)   # PMI2 server — uuid=0 MPICH builds
    OMPI_ROOT_ENV=(OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1)

    echo ""
    echo "  --- Section 3.1: driver.sh — srun launcher identification ---"
    APP="${ROCSHMEM_SRC}/build-ompi-auto/tests/functional_tests/rocshmem_functional_tests"
    if [[ -x "$APP" ]]; then
        export LAUNCHER=srun ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ipc
        export OMPI_ALLOW_RUN_AS_ROOT=1 OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1
        export PATH="${OMPI_BIN}:${PATH}"
        run_driver "driver-slurm-srun" "$APP"
        unset LAUNCHER ROCSHMEM_TEST_UUID ROCSHMEM_BACKEND OMPI_ALLOW_RUN_AS_ROOT OMPI_ALLOW_RUN_AS_ROOT_CONFIRM
        export PATH="${PATH#${OMPI_BIN}:}"
    else
        echo "  SKIP: ompi-auto binary not found"
    fi

    echo ""
    echo "  --- Section 3.2: direct — srun, ipc, ompi builds ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "srun-${build}-ipc-uuid0" "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc "${OMPI_ROOT_ENV[@]}" -- pass
        run_direct "srun-${build}-ipc-uuid1" "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ipc "${OMPI_ROOT_ENV[@]}" -- pass
    done

    echo ""
    echo "  --- Section 3.3: direct — srun, ipc, mpich builds ---"
    export LD_LIBRARY_PATH="${SYSTEM_LIB}:${MPICH_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH#${SYSTEM_LIB}:${OMPI_LIB}:${UCX_LIB}:}"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-auto-ipc-uuid0" "$APP" "${SRUN_PMI2[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc -- pass || echo "  SKIP: mpich-auto binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-off/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-off-ipc-uuid0"  "$APP" "${SRUN_PMI2[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ipc -- fail || echo "  SKIP: mpich-off binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-auto-ipc-uuid1" "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ipc -- pass || echo "  SKIP: mpich-auto binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-off/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-off-ipc-uuid1"  "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ipc -- pass || echo "  SKIP: mpich-off binary not found"  # TcpBS+IPC, no MPI
    export LD_LIBRARY_PATH="${SYSTEM_LIB}:${OMPI_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH#${SYSTEM_LIB}:${MPICH_LIB}:${UCX_LIB}:}"

    echo ""
    echo "  --- Section 3.4: direct — srun, ro, ompi builds ---"
    for build in ompi-auto ompi-on; do
        APP="${ROCSHMEM_SRC}/build-${build}/tests/functional_tests/rocshmem_functional_tests"
        [[ -x "$APP" ]] || { echo "  SKIP: ${build} binary not found"; continue; }
        run_direct "srun-${build}-ro-uuid0"  "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  "${OMPI_ROOT_ENV[@]}" -- pass
        run_direct "srun-${build}-ro-uuid1"  "$APP" "${SRUN_PMIX[@]}" -- ROCSHMEM_TEST_UUID=1 ROCSHMEM_BACKEND=ro  "${OMPI_ROOT_ENV[@]}" -- pass
    done

    echo ""
    echo "  --- Section 3.5: direct — srun, ro, mpich builds ---"
    export LD_LIBRARY_PATH="${SYSTEM_LIB}:${MPICH_LIB}:${UCX_LIB}:${LD_LIBRARY_PATH#${SYSTEM_LIB}:${OMPI_LIB}:${UCX_LIB}:}"
    APP="${ROCSHMEM_SRC}/build-mpich-auto/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-auto-ro-uuid0"  "$APP" "${SRUN_PMI2[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  -- pass || echo "  SKIP: mpich-auto binary not found"
    APP="${ROCSHMEM_SRC}/build-mpich-off/tests/functional_tests/rocshmem_functional_tests"
    [[ -x "$APP" ]] && run_direct "srun-mpich-off-ro-uuid0"   "$APP" "${SRUN_PMI2[@]}" -- ROCSHMEM_TEST_UUID=0 ROCSHMEM_BACKEND=ro  -- fail || echo "  SKIP: mpich-off binary not found"
    # uuid=1 ro for both mpich builds: not run — pmix/pmi2 conflict (see table at top)
    export LD_LIBRARY_PATH="${LD_LIBRARY_PATH#${SYSTEM_LIB}:${MPICH_LIB}:${UCX_LIB}:}"
fi

# ==========================================================================
# Summary table
# ==========================================================================
echo ""
echo "========================================================================"
echo " Results"
echo "========================================================================"
printf "  %-50s  %s\n" "Test case" "Result"
printf "  %-50s  %s\n" "$(printf '%0.s-' {1..50})" "--------"
for entry in "${RESULTS[@]}"; do
    status="${entry%%|*}"
    label="${entry##*|}"
    printf "  %-50s  %s\n" "${label}" "${status}"
done
echo ""
total=${#RESULTS[@]}
incorrect=$(printf '%s\n' "${RESULTS[@]}" | { grep -c "^INCORRECT" || true; })
correct=$(( total - incorrect ))
echo "  ${correct}/${total} correct"
echo "========================================================================"
if [[ $OVERALL_RC -eq 0 ]]; then
    echo "  ALL CASES CORRECT"
else
    echo "  ${incorrect} CASE(S) INCORRECT — check logs in ${LOG_BASE}"
fi
echo "========================================================================"
exit $OVERALL_RC
