#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-tests-rocprofv3-hip-stream-display
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)
find_package(rocDecode)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(hip-stream-display-env "${PRELOAD_ENV}")

add_test(
    NAME rocprofv3-test-hip-stream-display-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --kernel-rename -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json pftrace
        --log-level env -- $<TARGET_FILE:transpose>)

set_tests_properties(
    rocprofv3-test-hip-stream-display-execute
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               ENVIRONMENT
               "${hip-stream-display-env}"
               FAIL_REGULAR_EXPRESSION
               "threw an exception"
               DISABLED
               $<NOT:$<TARGET_EXISTS:transpose>>)

add_test(
    NAME rocprofv3-test-hip-stream-display-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-stream-display/out_results.json --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/hip-stream-display/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-hip-stream-display-validate
    PROPERTIES TIMEOUT
               45
               LABELS
               "integration-tests"
               DEPENDS
               rocprofv3-test-hip-stream-display-execute
               FAIL_REGULAR_EXPRESSION
               "AssertionError"
               DISABLED
               $<NOT:$<TARGET_EXISTS:transpose>>)
