#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-aborted-app
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# disable this test for thread sanitizers because of "signal-unsafe call inside signal"
# issues on mi200 and mi300 (works fine on vega20 and navi32)
if(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    set(IS_DISABLED ON)
else()
    # set(IS_DISABLED OFF)
    set(IS_DISABLED ON) # this test is currently unstable so we are disabling it
    # unconditionally for now
endif()

# app-abort
rocprofiler_add_integration_execute_test(
    rocprofv3-test-app-abort
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/out-aborted-app -- $<TARGET_FILE:vector-ops> 1 1
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROF_TESTING_RAISE_SIGNAL=1"
                "ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT=1"
    FIXTURES_SETUP rocprofv3-test-app-abort)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-app-abort
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini input.json
    TIMEOUT 45
    LABELS "integration-tests"
    DISABLED ${IS_DISABLED}
    FIXTURES_REQUIRED rocprofv3-test-app-abort
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/out-aborted-app/out_results.json)
