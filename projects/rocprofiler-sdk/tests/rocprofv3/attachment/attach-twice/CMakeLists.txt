#
# rocprofv3 attachment test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-attachment-attach-twice
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(ROCPROFILER_MEMCHECK_TYPES "AddressSanitizer" "UndefinedBehaviorSanitizer"
                               "ThreadSanitizer")

if(ROCPROFILER_MEMCHECK AND ROCPROFILER_MEMCHECK IN_LIST ROCPROFILER_MEMCHECK_TYPES)
    set(IS_DISABLED ON)
else()
    set(IS_DISABLED OFF)
endif()

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

set(attachment-env
    "${PRELOAD_ENV}"
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$ENV{LD_LIBRARY_PATH}"
    )

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

# Test that launches the app and reattaches to it twice (CSV format)
add_test(
    NAME rocprofv3-test-attachment-attach-twice-execute
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_test_unified.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} out)

set_tests_properties(
    rocprofv3-test-attachment-attach-twice-execute
    PROPERTIES TIMEOUT
               120
               LABELS
               "integration-tests;attachment"
               ENVIRONMENT
               "${attachment-env}"
               FAIL_REGULAR_EXPRESSION
               "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
               FIXTURES_SETUP
               rocprofv3-test-attachment-attach-twice
               DISABLED
               "${IS_DISABLED}")

# Validate the output from the reattached profiling (CSV)
add_test(
    NAME rocprofv3-test-attachment-attach-twice-csv-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_kernel_trace.csv --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_hsa_api_trace.csv
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_memory_copy_trace.csv
        --agent-input ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_agent_info.csv)

set_tests_properties(
    rocprofv3-test-attachment-attach-twice-csv-validate
    PROPERTIES TIMEOUT
               30
               LABELS
               "integration-tests;attachment"
               DEPENDS
               rocprofv3-test-attachment-attach-twice-execute
               FIXTURES_REQUIRED
               rocprofv3-test-attachment-attach-twice
               DISABLED
               "${IS_DISABLED}")

add_test(
    NAME rocprofv3-test-attachment-attach-twice-json-validate
    COMMAND
        ${Python3_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/validate.py --kernel-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json --hsa-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json
        --memory-copy-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json --agent-input
        ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json)

set_tests_properties(
    rocprofv3-test-attachment-attach-twice-json-validate
    PROPERTIES TIMEOUT
               30
               LABELS
               "integration-tests;attachment"
               DEPENDS
               rocprofv3-test-attachment-attach-twice-execute
               FIXTURES_REQUIRED
               rocprofv3-test-attachment-attach-twice
               DISABLED
               "${IS_DISABLED}")
