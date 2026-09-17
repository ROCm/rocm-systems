#
# rocprofv3 tool test for kernel replay (counter collection via snapshot/restore replay)
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-kernel-replay
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-generate-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

rocprofiler_configure_pytest_files(COPY test_kernel_replay_cli.py)

# Unit test for the rocprofv3 kernel-replay CLI logic. Imports rocprofv3.py directly, so
# it needs no GPU, no application run and no fixtures -- it guards which services replay
# rejects and how counter groups and input-file jobs turn into application runs. Run as a
# plain script rather than under pytest because conftest.py in this directory requires the
# integration-test arguments (--json-input, --passes).
add_test(
    NAME rocprofv3-test-kernel-replay-cli-unit
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_kernel_replay_cli.py
            --script $<TARGET_FILE:rocprofiler-sdk::rocprofv3>)

set_tests_properties(
    rocprofv3-test-kernel-replay-cli-unit
    PROPERTIES TIMEOUT 60 LABELS "unit-tests;kernel-replay" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Collect counters via in-process kernel replay. The number of --pmc groups drives the
# number of replay passes (one batch per pass), with device memory restored between
# passes. Each group shares SQ_WAVES and SQ_INSTS_VALU (the sanity counters that must be
# constant across a kernel's passes) plus one unique counter (so the batches are genuinely
# distinct). The kernel-replay app verifies its own results, so a broken restore fails
# this generate step. The execute helper serializes GPU work so other tests cannot
# contaminate the per-pass counters or performance baseline.
set(REPLAY_PASSES 5)

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --pmc SQ_WAVES SQ_INSTS_VALU
        SQ_INSTS_SALU --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM --pmc SQ_WAVES
        SQ_INSTS_VALU SQ_INSTS_LDS --replay-mode kernel --kernel-replay-beta-enabled
        --output-format csv json -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-data)

# Validate replay structure: each dispatch replayed REPLAY_PASSES times, the shared sanity
# counters (SQ_WAVES, SQ_INSTS_VALU) are constant across a dispatch's passes, and the
# kernels are present.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.json --passes
         ${REPLAY_PASSES} --common-counters SQ_WAVES SQ_INSTS_VALU -x
    TIMEOUT 120
    LABELS "kernel-replay"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-data)

# Regression test: replay must ignore ROCPROF_COUNTER_GROUPS_INTERVAL (which only paces
# the non-replay round-robin). With it set to 2, a replay path that reused that rotation
# would skip/duplicate groups. replay must still collect every group exactly once per
# dispatch.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-interval-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --pmc SQ_WAVES SQ_INSTS_VALU
        SQ_INSTS_SALU --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM --pmc SQ_WAVES
        SQ_INSTS_VALU SQ_INSTS_LDS --replay-mode kernel --kernel-replay-beta-enabled
        --output-format csv json -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval -o
        out -- $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay"
    ENVIRONMENT "${rocprofv3-generate-env};ROCPROF_COUNTER_GROUPS_INTERVAL=2"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-interval-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-interval-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval/out_results.json
         --passes ${REPLAY_PASSES} --common-counters SQ_WAVES SQ_INSTS_VALU -x
    TIMEOUT 120
    LABELS "kernel-replay"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-interval-data)

# One-pass baseline for performance scaling checks (same workload, single --pmc group).
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-baseline-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU
        --replay-mode kernel --kernel-replay-beta-enabled --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-baseline -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay;kernel-replay-perf"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-perf-data)

# Performance regression: total tool duration and per-pass spans must stay within the
# Figure 5 cost-model ceiling; 5-pass run must not blow up vs the 1-pass baseline.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-perf-validate BUNDLE_TESTS
    TEST_PATHS validate_perf.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.json
         --passes
         ${REPLAY_PASSES}
         --n-elems
         1048576
         --baseline-json
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-baseline/out_results.json
         --max-scaling-ratio
         10.0
    TIMEOUT 60
    LABELS "kernel-replay;kernel-replay-perf"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-data
                      rocprofv3-test-kernel-replay-perf-data)
