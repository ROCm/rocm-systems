#
# rocprofv3 attachment test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(attachment-env "${PRELOAD_ENV}")

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

# Test that launches the app and attaches to it (CSV format)
add_test(
    NAME rocprofv3-test-attachment-execute
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_test_unified.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} csv out)

set_tests_properties(
    rocprofv3-test-attachment-execute
    PROPERTIES TIMEOUT 60 LABELS "integration-tests" ENVIRONMENT "${attachment-env}"
               FAIL_REGULAR_EXPRESSION "failed to retrieve stream ID|ERROR|FATAL")

# Test that launches the app and attaches to it (JSON format)
add_test(
    NAME rocprofv3-test-attachment-json-execute
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_test_unified.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} json out_json)

set_tests_properties(
    rocprofv3-test-attachment-json-execute
    PROPERTIES TIMEOUT 60 LABELS "integration-tests" ENVIRONMENT "${attachment-env}"
               FAIL_REGULAR_EXPRESSION "failed to retrieve stream ID|ERROR|FATAL")

# Validate the output from the attached profiling (CSV)
add_test(
    NAME rocprofv3-test-attachment-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_kernel_trace.csv --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_hsa_api_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_memory_copy_trace.csv
        --agent-input ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-attachment-validate
    PROPERTIES TIMEOUT 30 LABELS "integration-tests" DEPENDS
               rocprofv3-test-attachment-execute)

# Validate the output from the attached profiling (JSON)
add_test(
    NAME rocprofv3-test-attachment-json-validate
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --json-input
            ${CMAKE_CURRENT_BINARY_DIR}/attachment-json-output/out_json_results.json)

set_tests_properties(
    rocprofv3-test-attachment-json-validate
    PROPERTIES TIMEOUT 30 LABELS "integration-tests" DEPENDS
               rocprofv3-test-attachment-json-execute)
