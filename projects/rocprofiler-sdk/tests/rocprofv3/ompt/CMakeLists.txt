#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-ompt-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread)Sanitizer" OR ROCPROFILER_BUILD_CODECOV)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

set(ROCPROFILER_MEMCHECK_TYPES "ThreadSanitizer" "AddressSanitizer"
                               "UndefinedBehaviorSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

# disable when GPU-0 is navi2, navi3, or navi4 (no OpenMP target offload support)
list(GET rocprofiler-sdk-tests-gfx-info 0 ompt-tracing-gpu-0-gfx-info)
if("${ompt-tracing-gpu-0-gfx-info}" MATCHES "^gfx(10|11|12)[0-9][0-9]$")
    set(IS_DISABLED ON)
endif()

# the openmp-target binary is what we trace; if it was not built, disable the test
if(NOT TARGET openmp-target)
    set(IS_DISABLED ON)
endif()

# Host-side OMPT tests trace the host-only openmp-host workload, which does not require
# target-offload support (so they are *not* gated on the Navi check). They are only
# disabled under sanitizers or when the host workload is missing.
if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_HOST_DISABLED ON)
else()
    set(IS_HOST_DISABLED OFF)
endif()

if(NOT TARGET openmp-host)
    set(IS_HOST_DISABLED ON)
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(ompt-tracing-env
    "OMP_NUM_THREADS=2"
    "OMP_TARGET_OFFLOAD=mandatory"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

# Host-only env
set(ompt-host-env
    "OMP_NUM_THREADS=2"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

set(ompt-tracing-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-trace)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace --hip-trace --kernel-trace
        -d ${ompt-tracing-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 240
    LABELS "integration-tests;openmp-target"
    ENVIRONMENT "${ompt-tracing-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-tracing
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k
         "not test_granular and not test_ompt_all_form_is_complete and not test_sys_trace"
         --rocpd-input ${ompt-tracing-output-dir}/out_results.db
    DISCOVERY_ARGS
        ${PYTEST_ARGS} -k
        "not test_granular and not test_ompt_all_form_is_complete and not test_sys_trace"
    TIMEOUT 45
    LABELS "integration-tests;openmp-target"
    FIXTURES_REQUIRED rocprofv3-test-ompt-tracing
    DISABLED "${IS_DISABLED}")

# Granular-target test: both OMPT execute tests use the same openmp-target binary so %tag%
# would collide. Use a literal output directory that matches what the validate test reads
# from.
set(ompt-granular-target-output-dir
    ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-granular-target-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-granular-target
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace=target -d
        ${ompt-granular-target-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target"
    ENVIRONMENT "${ompt-tracing-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-granular-target
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-granular-target
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k test_granular_target_filter_only_target_ops --rocpd-input
         ${ompt-granular-target-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_granular_target_filter_only_target_ops
    TIMEOUT 45
    LABELS "integration-tests;openmp-target"
    FIXTURES_REQUIRED rocprofv3-test-ompt-granular-target
    DISABLED "${IS_DISABLED}")

# Default --ompt-trace on a host-only OpenMP program: asserts the parallel region /
# implicit-task / work events a user expects to see are recorded.
set(ompt-host-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-host-trace)
set(ompt-host-validate-k
    "test_ompt_host_side_events_present or test_rocpd_contains_ompt_records or test_agent_info"
    )
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-host
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace -d ${ompt-host-output-dir}
        -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-host>,$<TARGET_FILE:openmp-host>,openmp-host>
    DEPENDS openmp-host
    TIMEOUT 100
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-host-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-host
    DISABLED "${IS_HOST_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-host
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k "${ompt-host-validate-k}" --rocpd-input
         ${ompt-host-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k "${ompt-host-validate-k}"
    TIMEOUT 45
    LABELS "integration-tests;openmp-host"
    FIXTURES_REQUIRED rocprofv3-test-ompt-host
    DISABLED "${IS_HOST_DISABLED}")

# Host-side granular filter: --ompt-trace parallel must record only the parallel-category
# operations (no task/sync/mutex/target leakage).
set(ompt-host-parallel-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-host-parallel-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-host-parallel
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace=parallel -d
        ${ompt-host-parallel-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-host>,$<TARGET_FILE:openmp-host>,openmp-host>
    DEPENDS openmp-host
    TIMEOUT 100
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-host-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-host-parallel
    DISABLED "${IS_HOST_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-host-parallel
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k test_granular_host_filter_only_parallel_ops --rocpd-input
         ${ompt-host-parallel-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_granular_host_filter_only_parallel_ops
    TIMEOUT 45
    LABELS "integration-tests;openmp-host"
    FIXTURES_REQUIRED rocprofv3-test-ompt-host-parallel
    DISABLED "${IS_HOST_DISABLED}")

# Explicit "trace everything" form
set(ompt-all-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-target-all-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-all
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace=all -d
        ${ompt-all-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-target>,$<TARGET_FILE:openmp-target>,openmp-target>
    DEPENDS openmp-target
    TIMEOUT 100
    LABELS "integration-tests;openmp-target"
    ENVIRONMENT "${ompt-tracing-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-all
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-all
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k test_ompt_all_form_is_complete --rocpd-input
         ${ompt-all-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_ompt_all_form_is_complete
    TIMEOUT 45
    LABELS "integration-tests;openmp-target"
    FIXTURES_REQUIRED rocprofv3-test-ompt-all
    DISABLED "${IS_DISABLED}")

# --sys-trace implicit OMPT: profiling with --sys-trace (no explicit --ompt-trace) and a
# non-rocpd --output-format must still implicitly enable OMPT and auto-add the rocpd
# output so OMPT records are captured. Uses --output-format csv to exercise the auto-add
# path. Runs on the host workload so it is not gated on target-offload support.
set(ompt-sys-trace-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-host-sys-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-sys-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${ompt-sys-trace-output-dir} -o out --output-format csv --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-host>,$<TARGET_FILE:openmp-host>,openmp-host>
    DEPENDS openmp-host
    TIMEOUT 100
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-host-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-sys-trace
    DISABLED "${IS_HOST_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-sys-trace
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k test_sys_trace_implicit_ompt_in_rocpd --rocpd-input
         ${ompt-sys-trace-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_sys_trace_implicit_ompt_in_rocpd
    TIMEOUT 45
    LABELS "integration-tests;openmp-host"
    FIXTURES_REQUIRED rocprofv3-test-ompt-sys-trace
    DISABLED "${IS_HOST_DISABLED}")
