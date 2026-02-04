#
# Late-start tracing integration test
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(rocprofiler-sdk-tests-late-start-tracing LANGUAGES CXX HIP)

# Set up environment for test execution
set(late-start-tracing-env
    "LD_LIBRARY_PATH=$<TARGET_FILE_DIR:rocprofiler-sdk::rocprofiler-sdk-shared-library>:$<TARGET_FILE_DIR:rocprofiler-sdk-late-start-tool>:$ENV{LD_LIBRARY_PATH}"
    )

# Execute late-start test - generates JSON output
rocprofiler_add_integration_execute_test(
    late-start-tracing
    COMMAND $<TARGET_FILE:late-start-tracing>
    DEPENDS rocprofiler-sdk-late-start-tool
    TIMEOUT 120
    LABELS "integration-tests;late-start"
    PRELOAD "${ROCPROFILER_MEMCHECK_PRELOAD_ENV_VALUE}"
    ENVIRONMENT "${late-start-tracing-env}"
                "ROCPROFILER_TOOL_OUTPUT_FILE=late-start-tracing.json"
    SKIP_REGULAR_EXPRESSION "test skipped"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_SETUP late-start-tracing)

# Validate JSON output using pytest
rocprofiler_add_integration_validate_test(
    late-start-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    TIMEOUT 45
    LABELS "integration-tests;late-start"
    FAIL_REGULAR_EXPRESSION "${ROCPROFILER_DEFAULT_FAIL_REGEX}"
    FIXTURES_REQUIRED late-start-tracing
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/late-start-tracing.json)
