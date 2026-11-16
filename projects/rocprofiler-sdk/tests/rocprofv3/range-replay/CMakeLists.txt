#
# rocprofv3 integration tests for range replay.
#
# The application opens one real range (begin/end, or the scoped host macros). rocprofv3
# collects two counter groups across that range. The validator requires every launch inside
# the range to carry both passes.
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-rocprofv3-range-replay LANGUAGES CXX)

find_package(rocprofiler-sdk REQUIRED)
find_package(Python3 REQUIRED)

set(rocprofv3-generate-env
    "PYTHONPATH=${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    )
set(REPLAY_PASSES 2)
set(RANGE_DISPATCHES 3)

function(add_rr_v3_replay_test MODE)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-range-replay-${MODE}-generate
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU --pmc
            SQ_WAVES GRBM_COUNT --replay-mode ${MODE} --kernel-replay-beta-enabled
            --output-format json -d ${CMAKE_CURRENT_BINARY_DIR}/${MODE} -o out --
            $<TARGET_FILE:range-replay-rocprofv3-app>
        DEPENDS range-replay-rocprofv3-app
        TIMEOUT 180
        LABELS "integration-tests;range-replay"
        ENVIRONMENT "${rocprofv3-generate-env};RR_V3_MODE=${MODE}"
        PASS_REGULAR_EXPRESSION "\\[rr-v3\\] PASS"
        FAIL_REGULAR_EXPRESSION "\\[rr-v3\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
        FIXTURES_SETUP rocprofv3-test-range-replay-${MODE}-data)

    rocprofiler_add_integration_validate_test(
        rocprofv3-test-range-replay-${MODE}-validate BUNDLE_TESTS
        TEST_PATHS validate.py
        COPY conftest.py
        ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/${MODE}/out_results.json --passes
             ${REPLAY_PASSES} --dispatches ${RANGE_DISPATCHES}
        TIMEOUT 60
        LABELS "integration-tests;range-replay"
        PYTHON_PATH
            "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
        FIXTURES_REQUIRED rocprofv3-test-range-replay-${MODE}-data)
endfunction()

# Packet replay of the range the application opened with begin/end.
add_rr_v3_replay_test(range)

# Host replay of the same dispatches, opened with the scoped macros. The tool's mode name
# is range-host; the application selects that path with RR_V3_MODE=host.
rocprofiler_add_integration_execute_test(
    rocprofv3-test-range-replay-range-host-generate
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --pmc SQ_WAVES SQ_INSTS_VALU --pmc
        SQ_WAVES GRBM_COUNT --replay-mode range-host --kernel-replay-beta-enabled
        --output-format json -d ${CMAKE_CURRENT_BINARY_DIR}/range-host -o out --
        $<TARGET_FILE:range-replay-rocprofv3-app>
    DEPENDS range-replay-rocprofv3-app
    TIMEOUT 180
    LABELS "integration-tests;range-replay"
    ENVIRONMENT "${rocprofv3-generate-env};RR_V3_MODE=host"
    PASS_REGULAR_EXPRESSION "\\[rr-v3\\] PASS"
    FAIL_REGULAR_EXPRESSION "\\[rr-v3\\] FAIL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP rocprofv3-test-range-replay-range-host-data)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-range-replay-range-host-validate BUNDLE_TESTS
    TEST_PATHS validate.py
    COPY conftest.py
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/range-host/out_results.json --passes
         ${REPLAY_PASSES} --dispatches ${RANGE_DISPATCHES}
    TIMEOUT 60
    LABELS "integration-tests;range-replay"
    PYTHON_PATH
        "${rocprofiler-sdk_LIB_DIR}/python${Python3_VERSION_MAJOR}.${Python3_VERSION_MINOR}/site-packages"
    FIXTURES_REQUIRED rocprofv3-test-range-replay-range-host-data)
