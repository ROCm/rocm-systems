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
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

set(rocprofv3-validate-env
    "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}"
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )

rocprofiler_configure_pytest_files(
    CONFIG pytest.ini
    COPY conftest.py validate.py validate_perf.py test_kernel_replay_cli.py
         input-job-replay-counters.json input-job-replay-att.json)

# Unit test for the rocprofv3 kernel-replay CLI logic. Imports rocprofv3.py directly, so
# it needs no GPU, no application run and no fixtures -- it guards which services replay
# rejects and how counter groups and input-file jobs turn into application runs. Run as a
# plain script rather than under pytest because conftest.py in this directory requires the
# integration-test arguments (--json-input, --passes).
add_test(
    NAME rocprofv3-test-kernel-replay-cli-unit
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/test_kernel_replay_cli.py
        --script $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --schema
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../source/docs/rocprofv3_input_schema.json)

set_tests_properties(
    rocprofv3-test-kernel-replay-cli-unit
    PROPERTIES TIMEOUT 60 LABELS "unit-tests;kernel-replay" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Collect counters via in-process kernel replay. The number of --pmc groups drives the
# number of replay passes (one batch per pass), with device memory restored between
# passes. Each group shares SQ_WAVES and SQ_INSTS_VALU (the sanity counters that must be
# constant across a kernel's passes) plus one unique counter (so the batches are genuinely
# distinct). The kernel-replay app verifies its own results, so a broken restore fails
# this generate step.
set(REPLAY_PASSES 5)

add_test(
    NAME rocprofv3-test-kernel-replay-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --pmc SQ_WAVES SQ_INSTS_VALU
        SQ_INSTS_SALU --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM --pmc SQ_WAVES
        SQ_INSTS_VALU SQ_INSTS_LDS --replay-mode kernel --kernel-replay-beta-enabled
        --output-format csv json -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-data)

# Validate replay structure: each dispatch replayed REPLAY_PASSES times, the shared sanity
# counters (SQ_WAVES, SQ_INSTS_VALU) are constant across a dispatch's passes, and the
# kernels are present.
add_test(
    NAME rocprofv3-test-kernel-replay-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.json --passes
        ${REPLAY_PASSES} --common-counters SQ_WAVES SQ_INSTS_VALU)

set_tests_properties(
    rocprofv3-test-kernel-replay-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Regression test: replay must ignore ROCPROF_COUNTER_GROUPS_INTERVAL (which only paces
# the non-replay round-robin). With it set to 2, a replay path that reused that rotation
# would skip/duplicate groups. replay must still collect every group exactly once per
# dispatch.
add_test(
    NAME rocprofv3-test-kernel-replay-interval-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --pmc SQ_WAVES SQ_INSTS_VALU
        SQ_INSTS_SALU --pmc SQ_WAVES SQ_INSTS_VALU SQ_INSTS_SMEM --pmc SQ_WAVES
        SQ_INSTS_VALU SQ_INSTS_LDS --replay-mode kernel --kernel-replay-beta-enabled
        --output-format csv json -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval -o
        out -- $<TARGET_FILE:kernel-replay> 1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-interval-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-generate-env};ROCPROF_COUNTER_GROUPS_INTERVAL=2"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-interval-data)

add_test(
    NAME rocprofv3-test-kernel-replay-interval-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-interval/out_results.json --passes
        ${REPLAY_PASSES} --common-counters SQ_WAVES SQ_INSTS_VALU)

set_tests_properties(
    rocprofv3-test-kernel-replay-interval-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-interval-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Regression test for AIPROFSDK-1130: --kernel-iteration-range selects logical launches,
# so replay passes must not consume iterations. With four launches, range 2-3, and two
# counter groups, exactly two logical dispatches must survive and each must carry passes
# [0, 1]. This exercises rejecting launch 1, accepting launches 2 and 3, rejecting launch
# 4, and resetting the cached decision between dispatches.
add_test(
    NAME rocprofv3-test-kernel-replay-iteration-range-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT
        --pmc SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel
        --kernel-replay-beta-enabled --kernel-include-regex saxpy
        --kernel-iteration-range "2-3" --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-iteration-range -o out --
        $<TARGET_FILE:kernel-replay> 65536 4)

set_tests_properties(
    rocprofv3-test-kernel-replay-iteration-range-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-iteration-range-data)

# Validate the exact ticket invariant: two selected logical dispatches, each carrying two
# distinct counter-group passes. Numeric dispatch IDs are deliberately not pinned.
add_test(
    NAME rocprofv3-test-kernel-replay-iteration-range-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-iteration-range/out_results.json
        --passes 2 --expected-dispatch-count 2 --common-counters SQ_WAVES SQ_INSTS_VALU
        -k
        "dispatch_id_constant_across_replay_passes or expected_logical_dispatch_count or each_pass_collects_distinct_batch"
    )

set_tests_properties(
    rocprofv3-test-kernel-replay-iteration-range-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-iteration-range-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# Dispatch thread trace as a replay pass of its own. With --att, pass 0 traces each
# targeted dispatch with counter collection switched off, and the counter groups follow on
# passes 1..N with the trace switched off. Counter records must still number their groups
# 0..N-1, and each dispatch must be traced exactly once rather than once per pass.
find_library(
    kernel_replay_attdecoder_LIBRARY
    NAMES rocprof-trace-decoder
    HINTS ${ROCM_PATH}
    PATHS ${ROCM_PATH}
    PATH_SUFFIXES lib)

set(KERNEL_REPLAY_ATT_DISABLED ON)
if(kernel_replay_attdecoder_LIBRARY)
    cmake_path(GET kernel_replay_attdecoder_LIBRARY PARENT_PATH
               kernel_replay_attdecoder_LIB_DIR)
    set(KERNEL_REPLAY_ATT_DISABLED OFF)
endif()

add_test(
    NAME rocprofv3-test-kernel-replay-att-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${kernel_replay_attdecoder_LIB_DIR} --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc
        SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel
        --kernel-replay-beta-enabled --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1)

# RUN_SERIAL: the validate step compares SQ counters, which count every wave on the
# device, across each dispatch's passes, so GPU work from concurrent tests makes it fail
# intermittently. The rocprofiler_kernel_replay lock only serializes the replay tests with
# each other.
set_tests_properties(
    rocprofv3-test-kernel-replay-att-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               RUN_SERIAL
               ON
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-att-data
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

add_test(
    NAME rocprofv3-test-kernel-replay-att-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att/out_results.json --passes 2
        --common-counters SQ_WAVES SQ_INSTS_VALU --expected-counters SQ_WAVES
        SQ_INSTS_VALU GRBM_COUNT GRBM_GUI_ACTIVE --thread-trace)

set_tests_properties(
    rocprofv3-test-kernel-replay-att-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-att-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

# With --att and no --kernel-iteration-range, only each kernel's first launch is profiled.
# The app launches its three kernels several times each, and the HIP runtime adds one copy
# kernel for hipMemcpyToSymbol, so four dispatches are a kernel's first launch: the later
# launches must be neither traced nor replayed, and each profiled one still carries both
# counter passes.
add_test(
    NAME rocprofv3-test-kernel-replay-att-first-iteration-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --att --att-library-path
        ${kernel_replay_attdecoder_LIB_DIR} --pmc SQ_WAVES SQ_INSTS_VALU GRBM_COUNT --pmc
        SQ_WAVES SQ_INSTS_VALU GRBM_GUI_ACTIVE --replay-mode kernel
        --kernel-replay-beta-enabled --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att-first-iteration -o out --
        $<TARGET_FILE:kernel-replay> 65536 3)

set_tests_properties(
    rocprofv3-test-kernel-replay-att-first-iteration-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-att-first-iteration-data
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

add_test(
    NAME rocprofv3-test-kernel-replay-att-first-iteration-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-att-first-iteration/out_results.json
        --passes 2 --expected-dispatch-count 4 --common-counters SQ_WAVES SQ_INSTS_VALU
        --thread-trace -k
        "dispatch_id_constant_across_replay_passes or expected_logical_dispatch_count or each_pass_collects_distinct_batch or one_thread_trace_per_dispatch"
    )

set_tests_properties(
    rocprofv3-test-kernel-replay-att-first-iteration-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-att-first-iteration-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

# job_replay_mode: kernel. The input file's jobs are the passes of one run rather than
# runs of their own: two counter jobs must produce the same replay structure as two --pmc
# groups.
add_test(
    NAME rocprofv3-test-kernel-replay-jobs-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-job-replay-counters.json --output-format json
        -d ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-jobs -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-jobs-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-jobs-data)

add_test(
    NAME rocprofv3-test-kernel-replay-jobs-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-jobs/out_results.json --passes 2
        --common-counters SQ_WAVES SQ_INSTS_VALU --expected-counters SQ_WAVES
        SQ_INSTS_VALU GRBM_COUNT GRBM_GUI_ACTIVE)

set_tests_properties(
    rocprofv3-test-kernel-replay-jobs-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-jobs-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

# The thread trace job sits between the two counter jobs, so its pass runs second: the
# counter records must still number their groups 0..1 and each dispatch must be traced
# exactly once.
add_test(
    NAME rocprofv3-test-kernel-replay-jobs-att-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input-job-replay-att.json --att-library-path
        ${kernel_replay_attdecoder_LIB_DIR} --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-jobs-att -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-jobs-att-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-jobs-att-data
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

add_test(
    NAME rocprofv3-test-kernel-replay-jobs-att-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-jobs-att/out_results.json --passes 2
        --common-counters SQ_WAVES SQ_INSTS_VALU --expected-counters SQ_WAVES
        SQ_INSTS_VALU GRBM_COUNT GRBM_GUI_ACTIVE --thread-trace)

set_tests_properties(
    rocprofv3-test-kernel-replay-jobs-att-validate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;thread-trace"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               rocprofv3-test-kernel-replay-jobs-att-data
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               ${KERNEL_REPLAY_ATT_DISABLED})

# One-pass baseline for performance scaling checks (same workload, single --pmc group).
add_test(
    NAME rocprofv3-test-kernel-replay-baseline-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU
        --replay-mode kernel --kernel-replay-beta-enabled --output-format json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-baseline -o out --
        $<TARGET_FILE:kernel-replay> 1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-baseline-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay;kernel-replay-perf"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
               RESOURCE_LOCK
               rocprofiler_kernel_replay
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-kernel-replay-perf-data)

# Performance regression: total tool duration and per-pass spans must stay within the
# Figure 5 cost-model ceiling; 5-pass run must not blow up vs the 1-pass baseline.
add_test(
    NAME rocprofv3-test-kernel-replay-perf-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate_perf.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay/out_results.json --passes
        ${REPLAY_PASSES} --n-elems 1048576 --baseline-json
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay-baseline/out_results.json
        --max-scaling-ratio 10.0)

set_tests_properties(
    rocprofv3-test-kernel-replay-perf-validate
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests;kernel-replay;kernel-replay-perf"
               ENVIRONMENT
               "${rocprofv3-validate-env}"
               FIXTURES_REQUIRED
               "rocprofv3-test-kernel-replay-data;rocprofv3-test-kernel-replay-perf-data"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
