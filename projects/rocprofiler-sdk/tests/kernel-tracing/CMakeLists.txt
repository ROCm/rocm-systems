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
    ENVIRONMENT "${kernel-tracing-env}"
    PRELOAD "$<TARGET_FILE:rocprofiler-sdk-json-tool>")

rocprofiler_add_integration_validate_test(
    kernel-tracing
    COPY validate.py conftest.py
    CONFIG pytest.ini
    ARGS ${PYTEST_ARGS} --input ${CMAKE_CURRENT_BINARY_DIR}/kernel-tracing-test.json
    DISCOVERY_ARGS ${PYTEST_ARGS})
