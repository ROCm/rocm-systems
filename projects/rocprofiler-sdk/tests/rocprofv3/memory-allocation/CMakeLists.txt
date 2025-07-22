#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-memory-allocation-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(memory-allocation-tracing-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-memory-allocation-tracing-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --memory-allocation-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json otf2
        --log-level env -- $<TARGET_FILE:hsa-memory-allocation>)

set_tests_properties(
    rocprofv3-test-memory-allocation-tracing-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT
               "${memory-allocation-tracing-env}" FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}")

add_test(
    NAME rocprofv3-test-memory-allocation-tracing-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/hsa-memory-allocation-trace/out_results.json
        --otf2-input
        ${CMAKE_CURRENT_BINARY_DIR}/hsa-memory-allocation-trace/out_results.otf2)

set_tests_properties(
    rocprofv3-test-memory-allocation-tracing-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               rocprofv3-test-memory-allocation-tracing-execute FAIL_REGULAR_EXPRESSION
               "AssertionError")
