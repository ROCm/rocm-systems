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

find_package(rocprofiler-sdk REQUIRED)

# hsa multiqueue dependency test
rocprofiler_add_integration_execute_test(
    rocprofv3-test-hsa-multiqueue
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --hsa-trace --kernel-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%argt%-trace -o out --output-format json csv pftrace
        ${PRELOAD_ARGS} -- $<TARGET_FILE:multiqueue_testapp>
    DEPENDS multiqueue_testapp
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_SETUP rocprofv3-test-hsa-multiqueue
    FAIL_REGULAR_EXPRESSION "HSA_API|HIP_API")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-hsa-multiqueue
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --hsa-trace-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_hsa_api_trace.csv
         --kernel-trace-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_kernel_trace.csv
         --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_results.json
         --pftrace-input
         ${CMAKE_CURRENT_BINARY_DIR}/multiqueue_testapp-trace/out_results.pftrace
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-hsa-multiqueue)
