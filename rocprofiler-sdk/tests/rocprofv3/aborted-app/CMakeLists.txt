#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-aborted-app
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py
                                                          input.json)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

# disable this test for thread sanitizers because of "signal-unsafe call inside signal"
# issues on mi200 and mi300 (works fine on vega20 and navi32)
if(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    set(DISABLE_THIS_TEST ON)
else()
    # set(DISABLE_THIS_TEST OFF)
    set(DISABLE_THIS_TEST ON) # this test is currently unstable so we are disabling it
                              # unconditionally for now
endif()

set(aborted-app-env "${PRELOAD_ENV}" ROCPROF_TESTING_RAISE_SIGNAL=1
                    ROCPROF_INTERNAL_TEST_SIGNAL_HANDLER_VIA_EXIT=1)

# app-abort
add_test(
    NAME rocprofv3-test-execute-app-abort
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i
        ${CMAKE_CURRENT_BINARY_DIR}/input.json -d
        ${CMAKE_CURRENT_BINARY_DIR}/out-aborted-app -- $<TARGET_FILE:vector-ops> 1 1)

set_tests_properties(
    rocprofv3-test-execute-app-abort
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${aborted-app-env}"
               WILL_FAIL
               TRUE
               DISABLED
               "${DISABLE_THIS_TEST}")

add_test(
    NAME rocprofv3-test-validate-app-abort
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            ${CMAKE_CURRENT_BINARY_DIR}/out-aborted-app/out_results.json)

set_tests_properties(
    rocprofv3-test-validate-app-abort
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-execute-app-abort"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${DISABLE_THIS_TEST}")
