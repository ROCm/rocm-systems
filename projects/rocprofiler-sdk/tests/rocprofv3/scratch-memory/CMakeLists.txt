#
# rocprofv3 tool test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-rocprofv3-scratch-memory-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

rocprofiler_configure_pytest_files(CONFIG pytest.ini COPY validate.py conftest.py)

string(REPLACE "LD_PRELOAD=" "ROCPROF_PRELOAD=" PRELOAD_ENV
               "${ROCPROFILER_MEMCHECK_PRELOAD_ENV}")

foreach(_OUTPUT_FORMAT csv json)
    rocprofiler_add_integration_execute_test(
        rocprofv3-test-scratch-memory-tracing-${_OUTPUT_FORMAT}
        COMMAND
            $<TARGET_FILE:rocprofiler-sdk::rocprofv3> --scratch-memory-trace -d
            ${CMAKE_CURRENT_BINARY_DIR}/%tag%-trace -o out --output-format
            ${_OUTPUT_FORMAT} --log-level env -- $<TARGET_FILE:scratch-memory>
        DEPENDS scratch-memory
        TIMEOUT 45
        LABELS "integration-tests"
        PRELOAD "${PRELOAD_ENV}"
        FIXTURES_SETUP rocprofv3-test-scratch-memory-tracing-${_OUTPUT_FORMAT}
        DISABLED "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}")
endforeach()

rocprofiler_add_integration_validate_test(
    rocprofv3-test-scratch-memory-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    ARGS --json-input ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-trace/out_results.json
         --csv-input
         ${CMAKE_CURRENT_BINARY_DIR}/scratch-memory-trace/out_scratch_memory_trace.csv
    TIMEOUT 45
    LABELS "integration-tests"
    FIXTURES_REQUIRED rocprofv3-test-scratch-memory-tracing-csv
                      rocprofv3-test-scratch-memory-tracing-json
    DISABLED "${ROCPROFILER_DISABLE_UNSTABLE_CTESTS}")
