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
    set(IS_DISABLED OFF)
endif()

# app-abort
#
# The profiled app raises SIGINT mid-run (ROCPROF_TESTING_RAISE_SIGNAL=1). rocprofv3's
# signal handler hands finalization off to its worker thread (which flushes the profiling
# output) and then re-raises SIGINT, so the process is terminated by the signal. CTest
# reports a signal-terminated process as an exception that WILL_FAIL does NOT invert, and
# a SIGINT termination additionally interrupts the ctest run. The shell wrapper turns that
# signal-termination into a regular non-zero exit code (128+signo) via "|| exit $?", which
# WILL_FAIL inverts into a pass. (Using "|| exit $?" instead of "; exit $?" keeps the
# whole command a single bash argument; CMake would otherwise split on the semicolon.) A
# genuine handler deadlock makes the wrapper block, so it is still caught by the ctest
# TIMEOUT below (timeouts are not inverted by WILL_FAIL).
rocprofiler_add_integration_execute_test(
    rocprofv3-test-app-abort
    COMMAND
        bash -c
        "$<TARGET_FILE:rocprofiler-sdk::rocprofv3> -i ${CMAKE_CURRENT_BINARY_DIR}/input.json -d ${CMAKE_CURRENT_BINARY_DIR}/out-aborted-app -- $<TARGET_FILE:vector-ops> 1 1 || exit $?"
    DEPENDS vector-ops
    TIMEOUT 45
    LABELS "integration-tests"
    WILL_FAIL
    DISABLED ${IS_DISABLED}
    PRELOAD "${PRELOAD_ENV}"
    ENVIRONMENT "ROCPROF_TESTING_RAISE_SIGNAL=1"
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
