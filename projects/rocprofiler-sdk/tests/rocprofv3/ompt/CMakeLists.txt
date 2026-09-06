#
#
#
cmake_minimum_required(VERSION 3.24 FATAL_ERROR)

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

# Host-side granular filter: --ompt-trace task must record only the task-category
# operations (no parallel/sync/mutex/target leakage). Uses 4 threads so that tasks are
# scheduled across worker threads rather than run inline.
set(ompt-host-task-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-host-task-trace)
set(ompt-host-task-env
    "OMP_NUM_THREADS=4"
    "OMP_DISPLAY_ENV=1"
    "ROCR_VISIBLE_DEVICES=0"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-host-task
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --ompt-trace=task -d
        ${ompt-host-task-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-host>,$<TARGET_FILE:openmp-host>,openmp-host>
    DEPENDS openmp-host
    TIMEOUT 100
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-host-task-env}"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-ompt-host-task
    DISABLED "${IS_HOST_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-ompt-host-task
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} -k test_granular_host_filter_only_task_ops --rocpd-input
         ${ompt-host-task-output-dir}/out_results.db
    DISCOVERY_ARGS ${PYTEST_ARGS} -k test_granular_host_filter_only_task_ops
    TIMEOUT 45
    LABELS "integration-tests;openmp-host"
    FIXTURES_REQUIRED rocprofv3-test-ompt-host-task
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

# --sys-trace implicit OMPT: profiling with --sys-trace (no explicit --ompt-trace) must
# still enable OMPT tracing, so the rocpd output contains OMPT records. OMPT is a
# rocpd-only trace, so the rocpd output format is requested explicitly. Runs on the host
# workload so it is not gated on target-offload support.
set(ompt-sys-trace-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-host-sys-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-sys-trace
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${ompt-sys-trace-output-dir} -o out --output-format rocpd --log-level env --
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

# Regression test for the OMPT-init deadlock, exercised through rocprofv3. rocprofv3 loads
# the tool library (which provides the ompt_start_tool symbol), so libomp engages OMPT and
# rocprofiler-sdk performs client discovery. The openmp-init-deadlock workload links a
# library whose DT_INIT constructor calls into libomp, reproducing the conditions under
# which discovery previously self-deadlocked. Reaching main means discovery completed
# without re-entering libomp's init lock.
#
# The assertion is twofold: TIMEOUT guards against the deadlock (a hang fails the test),
# and PASS_REGULAR_EXPRESSION ensures the workload actually reached main (so an unrelated
# early exit cannot pass silently). This is a host-only OpenMP workload, so it is gated on
# OMP_TARGET_COMPILER and the workload target rather than on target-offload support.
set(IS_INIT_DEADLOCK_DISABLED OFF)
if(NOT TARGET openmp-init-deadlock
   OR NOT OMP_TARGET_COMPILER
   OR (ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES))
    set(IS_INIT_DEADLOCK_DISABLED ON)
endif()

set(ompt-init-deadlock-output-dir ${CMAKE_CURRENT_BINARY_DIR}/openmp-init-deadlock-trace)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-ompt-init-deadlock
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace -d
        ${ompt-init-deadlock-output-dir} -o out --output-format rocpd --log-level env --
        $<IF:$<TARGET_EXISTS:openmp-init-deadlock>,$<TARGET_FILE:openmp-init-deadlock>,openmp-init-deadlock>
    DEPENDS openmp-init-deadlock
    TIMEOUT 20
    LABELS "integration-tests;openmp-host"
    ENVIRONMENT "${ompt-host-env}"
    PRELOAD "${PRELOAD_ENV}"
    PASS_REGULAR_EXPRESSION "reached main: no OMPT-init deadlock"
    DISABLED "${IS_INIT_DEADLOCK_DISABLED}")
