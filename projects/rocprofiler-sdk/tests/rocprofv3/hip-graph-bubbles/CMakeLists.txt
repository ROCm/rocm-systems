#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hip-graph-bubbles
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(hip-graph-bubbles-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-hip-graph-bubbles-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> -d
        ${CMAKE_CURRENT_BINARY_DIR}/rocprofv3-hip-graph-bubbles -o out --output-format
        rocpd --log-level env --hip-runtime-trace --kernel-trace --pmc GRBM_COUNT --
        $<TARGET_FILE:hip-graph-bubbles> 2000 10)

set_tests_properties(
    rocprofv3-test-hip-graph-bubbles-execute
    PROPERTIES TIMEOUT
               60
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${hip-graph-bubbles-env}"
               FAIL_REGULAR_EXPRESSION
               "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               DISABLED
               $<NOT:$<TARGET_EXISTS:hip-graph-bubbles>>)

add_test(
    NAME rocprofv3-test-hip-graph-bubbles-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --input
        ${CMAKE_CURRENT_BINARY_DIR}/rocprofv3-hip-graph-bubbles/out_results.db)

set_tests_properties(
    rocprofv3-test-hip-graph-bubbles-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-hip-graph-bubbles-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               $<NOT:$<TARGET_EXISTS:hip-graph-bubbles>>)
