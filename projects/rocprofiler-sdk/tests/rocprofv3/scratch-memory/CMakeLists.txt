#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-scratch-memory-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(scratch-memory-tracing-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-scratch-memory-tracing-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --scratch-memory-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json --log-level
        env -- $<TARGET_FILE:scratch-memory>)

set_tests_properties(
    rocprofv3-test-scratch-memory-tracing-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${scratch-memory-tracing-env}" FAIL_REGULAR_EXPRESSION
               "threw an exception")

add_test(NAME rocprofv3-test-scratch-memory-tracing-validate
         COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
                 ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-trace/out_results.json)

set_tests_properties(
    rocprofv3-test-scratch-memory-tracing-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-scratch-memory-tracing-execute FAIL_REGULAR_EXPRESSION
               "AssertionError")
