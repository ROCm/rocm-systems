#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-counter-collection-backpressure
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

# Execute test - drives a deep dispatch backlog (high iteration count with infrequent
# synchronization) under counter collection. Counter collection serializes dispatches;
# without producer backpressure the host runs thousands of kernels ahead of GPU
# completion, oversubscribing the HWS runlist and wedging the command processor. The
# serializer now bounds the staged-ahead depth automatically, so this deep backlog must
# drain without hanging; the tight TIMEOUT fails the test if the hang regresses.
rocprofiler_add_integration_execute_test(
    counter-collection-backpressure
    TARGET reproducible-dispatch-count
    ARGS 8000 2 100 8000
    TIMEOUT 120
    LABELS "integration-tests"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    ENVIRONMENT
        "ROCPROFILER_TOOL_OUTPUT_FILE=counter-collection-backpressure-test.json"
        "ROCPROFILER_TOOL_CONTEXTS=COUNTER_COLLECTION" "ROCPROF_COUNTERS=SQ_WAVES_sum"
    FIXTURES_SETUP counter-collection-backpressure
    UNSTABLE)

# Validate test - confirms the run completed and produced counter records for the full
# dispatch backlog (i.e. it did not hang or drop work under backpressure).
rocprofiler_add_integration_validate_test(
    counter-collection-backpressure
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED counter-collection-backpressure
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/counter-collection-backpressure-test.json
    UNSTABLE)
