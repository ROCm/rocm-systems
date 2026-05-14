#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-memory-allocation-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

rocprofiler_add_integration_execute_test(
    rocprofv3-test-memory-allocation-tracing
    COMMAND
        $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --memory-allocation-trace -d
        ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format json otf2 csv
        --log-level env -- $<TARGET_FILE:hsa-memory-allocation>
    DEPENDS hsa-memory-allocation
    TIMEOUT 45
    LABELS "integration-tests"
    PRELOAD "${PRELOAD_ENV}"
    FIXTURES_SETUP rocprofv3-test-memory-allocation-tracing)

rocprofiler_add_integration_validate_test(
    rocprofv3-test-memory-allocation-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input
         ${CMAKE_CURRENT_BINARY_DIR}/hsa-memory-allocation-trace/out_results.json
         --otf2-input
         ${CMAKE_CURRENT_BINARY_DIR}/hsa-memory-allocation-trace/out_results.otf2
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/hsa-memory-allocation-trace/out_memory_allocation_trace.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-memory-allocation-tracing)
