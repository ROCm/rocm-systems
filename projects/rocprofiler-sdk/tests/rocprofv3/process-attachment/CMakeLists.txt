#
# rocprofv3 live process attachment integration tests (pytest execute + validate)
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-process-attachment
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocprofiler_sdk_pytest REQUIRED)

rocprofiler_sdk_pc_sampling_disabled(IS_PC_SAMPLING_DISABLED)

if(IS_DISABLED OR IS_PC_SAMPLING_DISABLED)
    set(_PC_SAMPLING_ATTACH_DISABLED ON)
else()
    set(_PC_SAMPLING_ATTACH_DISABLED OFF)
endif()

set(ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX
    "PC sampling unavailable|PC sampling configuration is not supported|PROCESS_ATTACHMENT_SKIP: PC sampling unavailable"
    CACHE STRING "regex for skipping PC sampling tests when not supported")
set(_PC_SAMPLING_SELECTED_REGIONS_VALIDATE_FILTER
    "test_json_results_exist or test_pc_sampling_host_trap_samples or test_selected_regions_markers_captured"
)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning")
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(ATTACH_ENV "ROCP_TOOL_ATTACH=1")

# Must match PTRACE_SKIP_MESSAGE in attachment_utils.py (not generic text in tracebacks).
set(PROCESS_ATTACHMENT_SKIP_REGEX "PROCESS_ATTACHMENT_SKIP: ptrace unavailable")

# PMC live attach: agent selection and rocpd detach must not abort the workload
set(PMC_ATTACH_FAIL_REGEX
    "Could not set agent to be profiled|FOREIGN KEY constraint failed|sqlite3_step failed")
set(_PMC_ROCPD_VALIDATE_FILTER
    "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_pmc_counters_captured or test_rocpd_foreign_key_integrity or test_rocpd_no_duplicate_kernel_timestamps"
)
set(_HIP_ROCPD_VALIDATE_FILTER
    "test_rocpd_database_exists or test_rocpd_kernels_captured or test_rocpd_hip_regions_captured or test_rocpd_no_duplicate_kernel_timestamps"
)
set(_HIP_ROCPD_MATRIX_VALIDATE_FILTER
    "test_rocpd_database_exists or test_rocpd_matrix_transpose_kernels_captured or test_rocpd_hip_regions_captured or test_rocpd_no_duplicate_kernel_timestamps"
)
# OpenMP offload uses HSA/runtime paths; rocpd may have kernels but no HIP API regions.
set(_OPENMP_ATTACH_VALIDATE_FILTER
    "test_rocpd_database_exists or test_rocpd_openmp_offload_kernels_captured or test_rocpd_no_duplicate_kernel_timestamps"
)

find_package(MPI)
if(MPI_FOUND AND MPIEXEC_EXECUTABLE AND TARGET mpi-simple-attach)
    set(_MPI_ATTACH_DISABLED OFF)
else()
    set(_MPI_ATTACH_DISABLED ON)
endif()

set(MPI_ATTACH_SKIP_REGEX "PROCESS_ATTACHMENT_SKIP: MPI unavailable")

if(TARGET openmp-attach)
    set(_OPENMP_ATTACH_DISABLED OFF)
else()
    set(_OPENMP_ATTACH_DISABLED ON)
endif()

rocprofiler_configure_pytest_files(
    CONFIG pytest.ini
    COPY test_execute_attachment.py attachment_utils.py conftest.py validate.py)

set(_PYTEST_EXECUTE_SCRIPT
    ${CMAKE_CURRENT_BINARY_DIR}/test_execute_attachment.py)

# Timed attach with HIP + kernel tracing and rocpd output
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-hip-rocpd
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_hip_rocpd
        --test-app=$<TARGET_FILE:attachment-test>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/hip-rocpd
    DEPENDS attachment-test rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 90
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-hip-rocpd
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-hip-rocpd
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "${_HIP_ROCPD_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-hip-rocpd
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/hip-rocpd/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/hip-rocpd/attachment-output/skipped)

# rocpd-only attach with synchronous output
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-rocpd-sync
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_rocpd_sync
        --test-app=$<TARGET_FILE:attachment-test>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/rocpd-sync
    DEPENDS attachment-test rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 90
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-rocpd-sync
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-rocpd-sync
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "test_rocpd_database_exists or test_rocpd_hip_regions_captured"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-rocpd-sync
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sync/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/rocpd-sync/attachment-output/skipped)

# sys-trace + CSV duration attach
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-sys-trace-csv
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_sys_trace_csv
        --test-app=$<TARGET_FILE:attachment-test>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/sys-trace-csv
    DEPENDS attachment-test rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 90
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-sys-trace-csv
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-sys-trace-csv
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "test_csv_output_exists or test_kernel_trace_captured"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-sys-trace-csv
    ARGS --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/sys-trace-csv/attachment-output/out_kernel_trace.csv
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/sys-trace-csv/attachment-output/skipped)

# PMC + rocpd smoke: light attach (passes on fixed ROCm in CI)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-pmc-rocpd
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_pmc_rocpd_smoke
        --test-app=$<TARGET_FILE:attachment-test>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd
    DEPENDS attachment-test rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 120
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-pmc-rocpd
    FAIL_REGULAR_EXPRESSION
        "ERROR|FATAL|${PMC_ATTACH_FAIL_REGEX}|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-pmc-rocpd
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 60
    LABELS "integration-tests;process-attachment"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "${_PMC_ROCPD_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-pmc-rocpd
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd/attachment-output/skipped)

# transpose + long PMC attach + rocpd (stress rocpd flush on detach)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-pmc-rocpd-transpose-long
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_pmc_rocpd_transpose_long
        --test-app=$<TARGET_FILE:transpose>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-transpose-long
    DEPENDS transpose rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 360
    LABELS "integration-tests;process-attachment;pmc-rocpd-long"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-pmc-rocpd-transpose-long
    FAIL_REGULAR_EXPRESSION
        "ERROR|FATAL|${PMC_ATTACH_FAIL_REGEX}|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-pmc-rocpd-transpose-long
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 90
    LABELS "integration-tests;process-attachment;pmc-rocpd-long"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "${_PMC_ROCPD_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-pmc-rocpd-transpose-long
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-transpose-long/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-transpose-long/attachment-output/skipped)

# PMC attach under high multi-thread GPU load (attachment-test, 8 threads)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-pmc-rocpd-multithread
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_pmc_rocpd_multithread
        --test-app=$<TARGET_FILE:attachment-test>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-multithread
    DEPENDS attachment-test rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 150
    LABELS "integration-tests;process-attachment;pmc-rocpd-multithread"
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-pmc-rocpd-multithread
    FAIL_REGULAR_EXPRESSION
        "ERROR|FATAL|${PMC_ATTACH_FAIL_REGEX}|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-pmc-rocpd-multithread
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 60
    LABELS "integration-tests;process-attachment;pmc-rocpd-multithread"
    DISABLED ${IS_DISABLED}
    DISCOVERY_ARGS -k "${_PMC_ROCPD_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-pmc-rocpd-multithread
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-multithread/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/pmc-rocpd-multithread/attachment-output/skipped)

# transpose + host-trap PC sampling + roctx selected-regions (JSON output)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-pc-sampling-transpose-selected-regions
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_pc_sampling_transpose_selected_regions
        --test-app=$<TARGET_FILE:transpose>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-transpose-selected-regions
    DEPENDS transpose rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 90
    LABELS "integration-tests;process-attachment;pc-sampling;selected-regions"
    DISABLED ${_PC_SAMPLING_ATTACH_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION
        "${PROCESS_ATTACHMENT_SKIP_REGEX}|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-pc-sampling-transpose-selected-regions
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-pc-sampling-transpose-selected-regions
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 60
    LABELS "integration-tests;process-attachment;pc-sampling;selected-regions"
    DISABLED ${_PC_SAMPLING_ATTACH_DISABLED}
    DISCOVERY_ARGS -k "${_PC_SAMPLING_SELECTED_REGIONS_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED|${ROCPROFV3_TESTS_PC_SAMPLING_SKIP_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-pc-sampling-transpose-selected-regions
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-transpose-selected-regions/attachment-output/out_results.json
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/pc-sampling-transpose-selected-regions/attachment-output/skipped)

# mpiexec + mpi-simple-attach (long-running matrixTranspose per rank)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-mpi-simple-transpose
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_mpi_simple_transpose
        --test-app=$<TARGET_FILE:mpi-simple-attach>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/mpi-simple-transpose
        --mpiexec=${MPIEXEC_EXECUTABLE}
        --mpi-numproc-flag=${MPIEXEC_NUMPROC_FLAG}
    DEPENDS mpi-simple-attach rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 180
    LABELS "integration-tests;process-attachment;mpi"
    DISABLED ${_MPI_ATTACH_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION
        "${PROCESS_ATTACHMENT_SKIP_REGEX}|${MPI_ATTACH_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-mpi-simple-transpose
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-mpi-simple-transpose
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;process-attachment;mpi"
    DISABLED ${_MPI_ATTACH_DISABLED}
    DISCOVERY_ARGS -k "${_HIP_ROCPD_MATRIX_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED|${MPI_ATTACH_SKIP_REGEX}"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-mpi-simple-transpose
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/mpi-simple-transpose/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/mpi-simple-transpose/attachment-output/skipped)

# openmp-attach: long-running OpenMP offload for live attach (stable PID)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-process-attachment-openmp-offload
    COMMAND
        ${ROCPSDK_PYTEST_EXECUTABLE}
        ${_PYTEST_EXECUTE_SCRIPT}::test_attach_openmp_offload
        --test-app=$<TARGET_FILE:openmp-attach>
        --rocprofv3=$<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        --rocprof-log-level=${LOG_LEVEL}
        --output-name=out
        --output-dir=${CMAKE_CURRENT_BINARY_DIR}/openmp-offload
    DEPENDS openmp-attach rocprofiler-sdk::rocprofv3
    WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
    TIMEOUT 180
    LABELS "integration-tests;process-attachment;openmp"
    DISABLED ${_OPENMP_ATTACH_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "${ATTACH_ENV}"
    SKIP_REGULAR_EXPRESSION "${PROCESS_ATTACHMENT_SKIP_REGEX}"
    FIXTURES_SETUP rocprofv3-test-process-attachment-openmp-offload
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-process-attachment-openmp-offload
    TEST_PATHS validate.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;process-attachment;openmp"
    DISABLED ${_OPENMP_ATTACH_DISABLED}
    DISCOVERY_ARGS -k "${_OPENMP_ATTACH_VALIDATE_FILTER}"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-process-attachment-openmp-offload
    ARGS --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-offload/attachment-output/out_results.db
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/openmp-offload/attachment-output/skipped)
