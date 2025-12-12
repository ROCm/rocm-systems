#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Execute test - runs multistream with counter collection enabled
rocprofiler_add_integration_execute_test(
    counter-collection
    TARGET multistream
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=counter-collection-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=COUNTER_COLLECTION" "ROCPROF_COUNTERS=SQ_WAVES_sum"
    FIXTURES_SETUP counter-collection
    UNSTABLE)

# Validate test - validates the counter collection JSON output
rocprofiler_add_integration_validate_test(
    counter-collection
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED counter-collection
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/counter-collection-test.json
    UNSTABLE)
