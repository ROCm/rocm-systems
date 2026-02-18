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

# Unconditionally disabled: test has known data race in tool
set(IS_DISABLED ON)

if(ROCPROFILER_MEMCHECK STREQUAL "LeakSanitizer")
    set(LOG_LEVEL "warning") # info produces memory leak
else()
    set(LOG_LEVEL "info")
endif()

set(PRELOAD_ENV "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}")

# Test that launches the app and reattaches to it twice (CSV format)
rocprofiler_add_integration_execute_test(
    rocprofv3-test-attachment-attach-twice
    COMMAND
        ${CMAKE_CURRENT_SOURCE_DIR}/run_attachment_test_unified.sh
        $<TARGET_FILE:attachment-test> $<TARGET_FILE:rocprofiler-sdk::rocprofv3>
        ${CMAKE_CURRENT_BINARY_DIR} ${LOG_LEVEL} out
    DEPENDS rocprofiler-sdk::rocprofv3
    TIMEOUT 120
    LABELS "integration-tests;attachment"
    PRELOAD "${PRELOAD_ENV}"
    FAIL_REGULAR_EXPRESSION "ERROR|FATAL|${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    SKIP_REGULAR_EXPRESSION "This test is skipped."
    FIXTURES_SETUP rocprofv3-test-attachment-attach-twice
    DISABLED "${IS_DISABLED}")

# Validate the output from the reattached profiling (CSV)
rocprofiler_add_integration_validate_test(
    rocprofv3-test-attachment-attach-twice-csv
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_kernel_trace.csv
         --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_hsa_api_trace.csv
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_memory_copy_trace.csv
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_agent_info.csv
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/skipped
    TIMEOUT 30
    LABELS "integration-tests;attachment"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-attachment-attach-twice
    DISABLED "${IS_DISABLED}")

rocprofiler_add_integration_validate_test(
    rocprofv3-test-attachment-attach-twice-json
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --kernel-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json
         --hsa-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json
         --memory-copy-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json
         --agent-input
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/out_results.json
         --skip-if
         ${CMAKE_CURRENT_BINARY_DIR}/attachment-output/skipped
    TIMEOUT 30
    LABELS "integration-tests;attachment"
    SKIP_REGULAR_EXPRESSION "SKIPPED"
    FIXTURES_REQUIRED rocprofv3-test-attachment-attach-twice
    DISABLED "${IS_DISABLED}")
