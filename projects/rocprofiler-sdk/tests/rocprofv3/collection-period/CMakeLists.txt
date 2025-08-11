#
# rocprofv3 tool tests for collection period
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-collection-period
    LANGUAGES CXX
    VERSION 0.0.0)

if(ROCPROFILER_MEMCHECK STREQUAL "ThreadSanitizer")
    set(IS_THREAD_SANITIZER ON)
else()
    set(IS_THREAD_SANITIZER OFF)
endif()

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(collection-period-env "${PRELOAD_ENV}" ROCPROF_COLLECTION_PERIOD_TESTING=true)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

##########################################################################################
#
#   Command line input
#
##########################################################################################

add_test(
    NAME rocprofv3-test-collection-period-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --runtime-trace --summary -d
        ${CMAKE_CURRENT_BINARY_DIR}/collection-period -o out --output-format csv json
        pftrace otf2 --log-level warning --collection-period 0:1:1 1:1:2 0.5:0.5:0
        --collection-period-unit sec -- $<TARGET_FILE:reproducible-runtime> 5000 4)

set_tests_properties(
    rocprofv3-test-collection-period-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${collection-period-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}") # Once stable, should be disabled
                                                         # via IS_THREAD_SANITIZER (thread
                                                         # sanitizer has too much
                                                         # overhead)

add_test(
    NAME rocprofv3-test-collection-period-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/collection-period/out_results.json --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/collection-period/out_results.pftrace --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/collection-period/out_results.otf2
        --collection-period-input
        ${CMAKE_CURRENT_BINARY_DIR}/collection-period/out_collection_periods.log)

set_tests_properties(
    rocprofv3-test-collection-period-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               "rocprofv3-test-collection-period-execute"
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}") # Once stable, should be disabled
                                                         # via IS_THREAD_SANITIZER (thread
                                                         # sanitizer has too much
                                                         # overhead)
