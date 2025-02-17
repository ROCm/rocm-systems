#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-threading-records
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

add_library(pthread_wrapper SHARED)
target_sources(pthread_wrapper PRIVATE pthread_wrapper.cpp)

add_test(
    NAME rocprofv3-test-internal-threading
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --sys-trace --pmc SQ_WAVES
        --preload $<TARGET_FILE:pthread_wrapper> -- $<TARGET_FILE:transpose> 1 15)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set_tests_properties(
    rocprofv3-test-internal-threading
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${PRELOAD_ENV}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")
