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
        --output-format csv json rocpd -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay -o out
        -- $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-data)

# Validate replay structure: each dispatch replayed REPLAY_PASSES times, the shared sanity
# counters (SQ_WAVES, SQ_INSTS_VALU) are constant across a dispatch's passes, the kernels
# are present, and rocpd's counters_collection averages the shared counters over the
# passes rather than summing them.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.json
         --passes
         ${REPLAY_PASSES}
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.db
         -x
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
        --output-format csv json rocpd -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay"
    ENVIRONMENT "${rocprofv3-generate-env};ROCPROF_COUNTER_GROUPS_INTERVAL=2"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-interval-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-interval-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval/out_results.json
         --passes
         ${REPLAY_PASSES}
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --rocpd-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval/out_results.db
         -x
    TIMEOUT 120
    LABELS "kernel-replay"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-interval-data)

# Regression test for AIPROFSDK-1130: --kernel-iteration-range selects logical launches,
# so replay passes must not consume iterations. With four launches, range 2-3, and two
# counter groups, exactly two logical dispatches must survive and each must carry passes
# [0, 1]. This exercises rejecting launch 1, accepting launches 2 and 3, rejecting launch
# 4, and resetting the cached decision between dispatches.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-iteration-range-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel
        --kernel-replay-beta-enabled --kernel-include-regex saxpy --kernel-iteration-range
        "2-3" --kernel-trace --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-iteration-range -o out --
        $<TARGET_FILE:kernel-replay> 65536 4
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-iteration-range-data)

# Validate the exact ticket invariant: two selected logical dispatches, each carrying two
# distinct counter-group passes, which are the second and third saxpy launches. Numeric
# dispatch IDs are deliberately not pinned.
rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-iteration-range-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-iteration-range/out_results.json
         --passes
         2
         --expected-dispatch-count
         2
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --selected-kernel
         saxpy
         --selected-launches
         2
         3
         -k
         "dispatch_id_constant_across_replay_passes or expected_logical_dispatch_count or each_pass_collects_distinct_batch or unselected_dispatches_run_once or iteration_range_selects_launches"
    TIMEOUT 120
    LABELS "kernel-replay"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-iteration-range-data)

# Under --selected-regions only the launches inside a region are profiled, so only they
# count toward --kernel-iteration-range, with replay as without it. The application
# launches saxpy 3 times before the region, 5 times inside it and twice after it, so range
# 2-3 must replay the second and third launches inside the region.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-selected-regions-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --selected-regions --pmc SQ_WAVES
        SQ_INSTS_VALU GRBM_COUNT --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE
        --replay-mode kernel --kernel-replay-beta-enabled --kernel-include-regex saxpy
        --kernel-iteration-range "2-3" --kernel-trace --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-selected-regions -o out --
        $<TARGET_FILE:kernel-replay-regions> 65536 3 5 2
    DEPENDS kernel-replay-regions
    TIMEOUT 120
    LABELS "kernel-replay;selected-regions"
    ENVIRONMENT "${rocprofv3-generate-env}"
    FIXTURES_SETUP rocprofv3-test-kernel-replay-selected-regions-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-selected-regions-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-selected-regions/out_results.json
         --passes
         2
         --expected-dispatch-count
         2
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --selected-kernel
         saxpy
         --selected-launches
         2
         3
         -k
         "dispatch_id_constant_across_replay_passes or expected_logical_dispatch_count or each_pass_collects_distinct_batch or unselected_dispatches_run_once or iteration_range_selects_launches"
    TIMEOUT 120
    LABELS "kernel-replay;selected-regions"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-selected-regions-data)

# Dispatch thread trace as the last replay pass. With --att, passes 0..N-1 collect the N
# counter groups with thread trace stopped, and pass N traces the dispatch with counters
# stopped. Counter records must still number their groups 0..N-1 with no row for the trace
# pass, and each dispatch must be traced exactly once rather than once per pass. Needs the
# trace decoder, like the other thread trace tests.
rocprofiler_sdk_find_trace_decoder()

set(KERNEL_REPLAY_ATT_DISABLED ON)
if(attdecoder_FOUND)
    set(KERNEL_REPLAY_ATT_DISABLED OFF)
endif()

rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-att-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${attdecoder_LIB_DIR} --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc SQ_WAVES
        SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel --kernel-replay-beta-enabled
        --output-format json -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay;thread-trace"
    ENVIRONMENT "${rocprofv3-generate-env}"
    DISABLED ${KERNEL_REPLAY_ATT_DISABLED}
    FIXTURES_SETUP rocprofv3-test-kernel-replay-att-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-att-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att/out_results.json
         --passes
         2
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --expected-counters
         SQ_WAVES
         SQ_INSTS_VALU
         GRBM_COUNT
         GRBM_GUI_ACTIVE
         --thread-trace
         -x
    TIMEOUT 120
    LABELS "kernel-replay;thread-trace"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    DISABLED ${KERNEL_REPLAY_ATT_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-att-data)

# With --att and no --kernel-iteration-range only each kernel's first launch is profiled.
# The app launches its three kernels three times each, and the HIP runtime adds one copy
# kernel for hipMemcpyToSymbol, so four dispatches are a kernel's first launch. The later
# launches must be neither traced nor replayed (one kernel-trace record each), and each
# profiled launch runs once per counter group plus once for thread trace.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-kernel-replay-att-first-iteration-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${attdecoder_LIB_DIR} --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc SQ_WAVES
        SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel --kernel-replay-beta-enabled
        --kernel-trace --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att-first-iteration -o out --
        $<TARGET_FILE:kernel-replay> 65536 3
    DEPENDS kernel-replay
    TIMEOUT 120
    LABELS "kernel-replay;thread-trace"
    ENVIRONMENT "${rocprofv3-generate-env}"
    DISABLED ${KERNEL_REPLAY_ATT_DISABLED}
    FIXTURES_SETUP rocprofv3-test-kernel-replay-att-first-iteration-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-kernel-replay-att-first-iteration-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att-first-iteration/out_results.json
         --passes
         2
         --expected-dispatch-count
         4
         --common-counters
         SQ_WAVES
         SQ_INSTS_VALU
         --thread-trace
         -k
         "dispatch_id_constant_across_replay_passes or expected_logical_dispatch_count or each_pass_collects_distinct_batch or one_thread_trace_per_dispatch or thread_trace_pass_has_no_counter_row or thread_trace_runs_one_extra_pass or unselected_dispatches_run_once or common_counters_constant_across_passes"
    TIMEOUT 120
    LABELS "kernel-replay;thread-trace"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    DISABLED ${KERNEL_REPLAY_ATT_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-kernel-replay-att-first-iteration-data)

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
