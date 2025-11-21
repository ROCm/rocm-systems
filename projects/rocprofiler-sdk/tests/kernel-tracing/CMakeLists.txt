#
#
#
cmake_minimum_required(VERSION 3.21.0 FATAL_ERROR)

project(
    rocprofiler-sdk-tests-kernel-tracing
    LANGUAGES CXX
    VERSION 0.0.0)

find_package(rocprofiler-sdk REQUIRED)

set(PYTEST_ARGS)
if(ROCPROFILER_MEMCHECK MATCHES "(Address|Thread|UndefinedBehavior)Sanitizer"
   OR ROCPROFILER_BUILD_CODECOV
   OR ROCPROFILER_DISABLE_UNSTABLE_CTESTS)
    set(PYTEST_ARGS -k "not test_total_runtime")
endif()

set(kernel-tracing-env "ROCPROFILER_TOOL_DISABLE_PERFETTO=1"
                       "ROCPROFILER_TOOL_OUTPUT_FILE=kernel-tracing-test.json")

rocprofiler_add_integration_execute_test(
    kernel-tracing
    TARGET reproducible-runtime
    TIMEOUT 100
    LABELS "integration-tests"
    ENVIRONMENT "${kernel-tracing-env}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>"
    FIXTURES_SETUP kernel-tracing)

rocprofiler_add_integration_validate_test(
    kernel-tracing
    TEST_PATHS validate.py
    COPY conftest.py
    CONFIG pytest.ini
    LABELS "integration-tests"
    TIMEOUT 45
    FIXTURES_REQUIRED kernel-tracing
    DISCOVERY_ARGS ${PYTEST_ARGS}
    ARGS --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json
    UNSTABLE)
