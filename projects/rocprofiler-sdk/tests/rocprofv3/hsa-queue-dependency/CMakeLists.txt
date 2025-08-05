#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-hsa-queue-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

string(REPLACE "LD_PRELOAD=" "--preload;" PRELOAD_ARGS
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(tracing-env "")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

find_package(rocprofiler-sdk REQUIRED)

# hsa multiqueue dependency test
add_test(
    NAME rocprofv3-test-hsa-multiqueue-execute
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out --output-format json csv pftrace
        ${PRELOAD_ARGS} -- $<TARGET_FILE:multiqueue_testapp>)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-execute
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" ENVIRONMENT "${tracing-env}"
               FAIL_REGULAR_EXPRESSION "HSA_API|HIP_API")

add_test(
    NAME rocprofv3-test-hsa-multiqueue-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --hsa-trace-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_hsa_api_trace.csv
        --kernel-trace-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_kernel_trace.csv
        --json-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_results.json
        --pftrace-input
        ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_results.pftrace)

set_tests_properties(
    rocprofv3-test-hsa-multiqueue-validate
    PROPERTIES TIMEOUT 45 LABELS "integration-tests" DEPENDS
               "rocprofv3-test-hsa-multiqueue-execute" FAIL_REGULAR_EXPRESSION
               "AssertionError")
