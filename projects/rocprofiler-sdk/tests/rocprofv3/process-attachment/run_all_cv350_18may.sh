#!/usr/bin/env bash
# Run all process-attachment execute + validate tests with install_18May rocprofv3.
set -uo pipefail

BUILD="${BUILD:-/home/rpathani/rocm-systems-build/rocprofiler-sdk-sr}"
PA="${BUILD}/tests/rocprofv3/process-attachment"
ROCPROFV3="${ROCPROFV3:-/home/rpathani/therock-tarball/install_18May/bin/rocprofv3}"
export LD_LIBRARY_PATH="/home/rpathani/therock-tarball/install_18May/lib:${LD_LIBRARY_PATH:-}"
export ROCP_TOOL_ATTACH=1

BIN="${BUILD}/bin"
ATTACH_TEST="${BIN}/attachment-test"
TRANSPOSE="${BIN}/transpose"
MPI_SIMPLE_ATTACH="${BIN}/mpi-simple-attach"
OPENMP_ATTACH="${BIN}/openmp-attach"
MPIEXEC="${MPIEXEC:-/opt/amd/rochpl/tpl/openmpi/bin/mpiexec}"

LOGDIR="${PA}/logs"
TS="$(date +%Y%m%d-%H%M%S)"
REPORT="${LOGDIR}/process-attachment-all-18may-${TS}.txt"
mkdir -p "$LOGDIR"
cd "$PA"

COMMON=(--rocprofv3="$ROCPROFV3" --rocprof-log-level=info --output-name=out)
PASS=0
FAIL=0
SKIP=0

log() { echo "$@" | tee -a "$REPORT"; }

run_pair() {
  local tag="$1"
  local pytest_name="$2"
  local validate_filter="$3"
  shift 3
  local -a exec_args=()
  local -a val_args=()
  local after_dash=0
  for a in "$@"; do
    if [ "$a" = "--" ]; then after_dash=1; continue; fi
    if [ "$after_dash" -eq 1 ]; then val_args+=("$a"); else exec_args+=("$a"); fi
  done

  log ""
  log "======== $tag ========"
  local t0 t1 ec outdir=""
  t0=$(date +%s)
  if pytest -vv -s test_execute_attachment.py::"$pytest_name" "${exec_args[@]}" \
    >"${LOGDIR}/${tag}-${TS}-execute.log" 2>&1; then
    ec=0
  else
    ec=$?
  fi
  t1=$(date +%s)
  log "execute exit=$ec duration=$((t1 - t0))s log=${LOGDIR}/${tag}-${TS}-execute.log"

  for i in "${!exec_args[@]}"; do
    if [ "${exec_args[$i]}" = "--output-dir" ]; then outdir="${exec_args[$i+1]:-}"; fi
  done

  if [ "$ec" -ne 0 ]; then
    if [ -n "$outdir" ] && [ -f "$outdir/attachment-output/pc-sampling-skipped" ]; then
      log "SKIP (pc-sampling unavailable)"
      SKIP=$((SKIP + 1))
      return 0
    fi
    FAIL=$((FAIL + 1))
    return 0
  fi

  if pytest -vv validate.py -k "$validate_filter" "${val_args[@]}" \
    >"${LOGDIR}/${tag}-${TS}-validate.log" 2>&1; then
    log "validate PASS log=${LOGDIR}/${tag}-${TS}-validate.log"
    PASS=$((PASS + 1))
  else
    log "validate FAIL log=${LOGDIR}/${tag}-${TS}-validate.log"
    FAIL=$((FAIL + 1))
  fi
}

log "Process-attachment full run install_18May"
log "rocprofv3=$ROCPROFV3"
log "report=$REPORT"

run_pair hip-rocpd test_attach_hip_rocpd \
  "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_hip_regions_captured or test_rocpd_no_duplicate_kernel_timestamps" \
  --test-app="$ATTACH_TEST" --output-dir="$PA/hip-rocpd" "${COMMON[@]}" \
  -- --rocpd-input="$PA/hip-rocpd/attachment-output/out_results.db" --skip-if="$PA/hip-rocpd/attachment-output/skipped"

run_pair rocpd-sync test_attach_rocpd_sync \
  "test_rocpd_database_exists or test_rocpd_hip_regions_captured" \
  --test-app="$ATTACH_TEST" --output-dir="$PA/rocpd-sync" "${COMMON[@]}" \
  -- --rocpd-input="$PA/rocpd-sync/attachment-output/out_results.db" --skip-if="$PA/rocpd-sync/attachment-output/skipped"

run_pair sys-trace-csv test_attach_sys_trace_csv \
  "test_csv_output_exists or test_kernel_trace_captured" \
  --test-app="$ATTACH_TEST" --output-dir="$PA/sys-trace-csv" "${COMMON[@]}" \
  -- --kernel-input="$PA/sys-trace-csv/attachment-output/out_kernel_trace.csv" --skip-if="$PA/sys-trace-csv/attachment-output/skipped"

run_pair pmc-rocpd test_attach_pmc_rocpd_smoke \
  "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_pmc_counters_captured or test_rocpd_foreign_key_integrity or test_rocpd_no_duplicate_kernel_timestamps" \
  --test-app="$ATTACH_TEST" --output-dir="$PA/pmc-rocpd" "${COMMON[@]}" \
  -- --rocpd-input="$PA/pmc-rocpd/attachment-output/out_results.db" --skip-if="$PA/pmc-rocpd/attachment-output/skipped"

run_pair pmc-multithread test_attach_pmc_rocpd_multithread \
  "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_pmc_counters_captured or test_rocpd_foreign_key_integrity or test_rocpd_no_duplicate_kernel_timestamps" \
  --test-app="$ATTACH_TEST" --output-dir="$PA/pmc-rocpd-multithread" "${COMMON[@]}" \
  -- --rocpd-input="$PA/pmc-rocpd-multithread/attachment-output/out_results.db" --skip-if="$PA/pmc-rocpd-multithread/attachment-output/skipped"

run_pair pmc-transpose-long test_attach_pmc_rocpd_transpose_long \
  "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_pmc_counters_captured or test_rocpd_foreign_key_integrity or test_rocpd_no_duplicate_kernel_timestamps" \
  --test-app="$TRANSPOSE" --output-dir="$PA/pmc-rocpd-transpose-long" "${COMMON[@]}" \
  -- --rocpd-input="$PA/pmc-rocpd-transpose-long/attachment-output/out_results.db" --skip-if="$PA/pmc-rocpd-transpose-long/attachment-output/skipped"

run_pair pc-sampling test_attach_pc_sampling_transpose_selected_regions \
  "test_json_results_exist or test_pc_sampling_host_trap_samples or test_selected_regions_markers_captured" \
  --test-app="$TRANSPOSE" --output-dir="$PA/pc-sampling-transpose-selected-regions" "${COMMON[@]}" \
  -- --json-input="$PA/pc-sampling-transpose-selected-regions/attachment-output/out_results.json" --skip-if="$PA/pc-sampling-transpose-selected-regions/attachment-output/skipped"

if [ -x "$MPIEXEC" ] && [ -x "$MPI_SIMPLE_ATTACH" ]; then
  run_pair mpi test_attach_mpi_simple_transpose \
    "test_rocpd_database_exists or test_rocpd_matrix_transpose_kernels_captured or test_rocpd_hip_regions_captured or test_rocpd_no_duplicate_kernel_timestamps" \
    --test-app="$MPI_SIMPLE_ATTACH" --output-dir="$PA/mpi-simple-transpose" \
    --mpiexec="$MPIEXEC" --mpi-numproc-flag=-n "${COMMON[@]}" \
    -- --rocpd-input="$PA/mpi-simple-transpose/attachment-output/out_results.db" --skip-if="$PA/mpi-simple-transpose/attachment-output/skipped"
else
  log "mpi SKIP: missing mpiexec or mpi-simple-attach"
  SKIP=$((SKIP + 1))
fi

if [ -x "$OPENMP_ATTACH" ]; then
  run_pair openmp test_attach_openmp_offload \
    "test_rocpd_database_exists or test_rocpd_openmp_offload_kernels_captured or test_rocpd_no_duplicate_kernel_timestamps" \
    --test-app="$OPENMP_ATTACH" --output-dir="$PA/openmp-offload" "${COMMON[@]}" \
    -- --rocpd-input="$PA/openmp-offload/attachment-output/out_results.db" --skip-if="$PA/openmp-offload/attachment-output/skipped"
else
  log "openmp SKIP: missing openmp-attach"
  SKIP=$((SKIP + 1))
fi

log ""
log "======== SUMMARY ========"
log "passed (execute+validate): $PASS"
log "failed: $FAIL"
log "skipped: $SKIP"
log "report: $REPORT"
