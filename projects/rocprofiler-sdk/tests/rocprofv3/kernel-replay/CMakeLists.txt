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

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY conftest.py validate.py)

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
        SQ_INSTS_VALU SQ_INSTS_LDS --kernel-replay --output-format csv json -d
        ${CMAKE_CURRENT_BINARY_DIR}/kernel-replay -o out -- $<TARGET_FILE:kernel-replay>
        1048576 1)

set_tests_properties(
    rocprofv3-test-kernel-replay-generate
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;kernel-replay"
               ENVIRONMENT
               "${rocprofv3-generate-env}"
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
